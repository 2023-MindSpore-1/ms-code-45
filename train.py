import os
# os.system('pip install gensim -i https://pypi.tuna.tsinghua.edu.cn/simple/')
# os.system('pip install ruamel.yaml -i https://pypi.tuna.tsinghua.edu.cn/simple/')
# os.system('pip install logzero -i https://pypi.tuna.tsinghua.edu.cn/simple/')
# os.system('pip install gensim -i https://pypi.tuna.tsinghua.edu.cn/simple/')
# os.system('pip install ruamel.yaml -i https://pypi.tuna.tsinghua.edu.cn/simple/')
# os.system('pip install logzero -i https://pypi.tuna.tsinghua.edu.cn/simple/')
import click
import numpy as np
import argparse
import ast
from pathlib import Path
from ruamel.yaml import YAML
from sklearn.model_selection import train_test_split
from logzero import logger
from mindspore import nn
from mindspore import Model
from mindspore import ops
from mindspore.profiler import Profiler
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import ModelCheckpoint, LossMonitor, TimeMonitor, CheckpointConfig
from deepxml.dataset import MultiLabelDataset
from deepxml.data_utils import get_data, get_mlb, get_word_emb, output_res

from deepxml.xmlcnn import XMLCNN, CorNetXMLCNN
from deepxml.trainonestep import XMLTrainOneStepCell
from deepxml.models import CoreModel
from deepxml.callback import EvalCallBack
import mindspore.dataset as ds
from mindspore import context
import mindspore as ms


# context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

parser = argparse.ArgumentParser(description='Train CorNet')

parser.add_argument('--run_modelarts', type=ast.literal_eval, default=False, help='train modelarts')
parser.add_argument('--is_distributed', type=ast.literal_eval, default=False, help="use 8 npus")
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--data_url', type=str, default='/opt_data/lh', help="root path to data directory")
parser.add_argument('--train_url', type=str, default='log')
parser.add_argument('--device_target', type=str, default='Ascend')
parser.add_argument('--train_valid', type=ast.literal_eval, default=True)
parser.add_argument('--dataset_path',
                    type=str,
                    default="/mnt/nvme1/deep_data",
                    help='path where the dataset is saved')
parser.add_argument('--save_checkpoint_path',
                    type=str,
                    default="./ckpt",
                    help='if is test, must provide\
                    path where the trained ckpt file')
args = parser.parse_args()

model_dict = {
        # 'AttentionXML': AttentionXML,
        # 'CorNetAttentionXML': CorNetAttentionXML,
        # 'MeSHProbeNet': MeSHProbeNet,
        # 'CorNetMeSHProbeNet': CorNetMeSHProbeNet,
        # 'BertXML': BertXML,
        # 'CorNetBertXML': CorNetBertXML,
        'XMLCNN': XMLCNN,
        'CorNetXMLCNN': CorNetXMLCNN,
        # 'CorNetXMLCNN': CorNetXMLCNN(dropout=0.5, labels_num=3801,dynamic_pool_length=8, bottleneck_dim=512, num_filters=128, vocab_size=500, embedding_size=300)
        }

class EvalNet(nn.Cell):
    def __init__(self, network, k):
        super(EvalNet, self).__init__()
        self.network = network
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.topk = ops.TopK()
        self.k = k
    def construct(self, x, y):
        logits = self.network(x)
        loss = self.loss_fn(logits, y)
        score, index = self.topk(logits, self.k)
        return loss, index

if __name__ == '__main__':
    yaml = YAML(typ='safe')
    if args.run_modelarts:
        import moxing as mox
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_cnf = os.path.join(current_dir, 'configure/datasets/EUR-Lex.yaml')
        model_cnf = os.path.join(current_dir,'configure/models/CorNetXMLCNN-EUR-Lex.yaml')
        data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
        obs_data_url = args.data_url
        args.data_url = '/home/work/user-job-dir/inputs/data/'
        obs_train_url = args.train_url
        args.train_url = '/home/work/user-job-dir/outputs/model/'
        try:
            mox.file.copy_parallel(obs_data_url, args.data_url)
            print("Successfully Download {} to {}".format(obs_data_url,
                                                          args.data_url))
        except Exception as e:
            print('moxing download {} to {} failed: '.format(
                obs_data_url, args.data_url) + str(e))
        args.dataset_path = args.data_url
        args.save_checkpoint_path = args.train_url
        data_cnf_embedding_emb_init = os.path.join(args.dataset_path,'deep_data/EUR-Lex/emb_init.npy')
        data_cnf_train_texts = os.path.join(args.dataset_path,'deep_data/EUR-Lex/train_texts.npy')
        data_cnf_train_labels = os.path.join(args.dataset_path, 'deep_data/EUR-Lex/train_labels.npy')
        data_cnf_labels_binarizer = os.path.join(args.dataset_path, 'deep_data/EUR-Lex/labels_binarizer')
    else:
        data_cnf = 'configure/datasets/EUR-Lex.yaml'
        model_cnf = 'configure/models/CorNetXMLCNN-EUR-Lex.yaml'
        data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
        # data_cnf_embedding_emb_init = data_cnf['embedding']['emb_init']
        # data_cnf_train_texts = data_cnf['train']['texts']
        # data_cnf_train_labels = data_cnf['train']['labels']
        # data_cnf_labels_binarizer = data_cnf['labels_binarizer']
        data_cnf_embedding_emb_init = os.path.join(args.dataset_path, './EUR-Lex/emb_init.npy')
        data_cnf_train_texts = os.path.join(args.dataset_path, './EUR-Lex/train_texts.npy')
        data_cnf_train_labels = os.path.join(args.dataset_path, 'EUR-Lex/train_labels.npy')
        data_cnf_labels_binarizer = os.path.join(args.dataset_path, './EUR-Lex/labels_binarizer')
        args.save_checkpoint_path = './'
    mode = None
    if args.is_distributed:
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE', '1'))
        context.set_context(device_id=device_id)
        ms.set_auto_parallel_context(device_num=device_num, parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                     gradients_mean=True)
        init()
        rank_id = get_rank()
        ckpt_save_dir = os.path.join(args.save_checkpoint_path, "ckpt_" + str(rank_id) + "/")
    else:
        device_id = int(os.getenv('DEVICE_ID','0'))
        context.set_context(device_id=device_id)
        rank_id = 0
        device_num = 1
        ckpt_save_dir = os.path.join(args.save_checkpoint_path,'./')
    # profiles = Profiler()


    model, model_name, data_name = None, model_cnf['name'], data_cnf['name']
    model_path = os.path.join(model_cnf['path'], F'{model_name}-{data_name}')
    emb_init = get_word_emb(data_cnf_embedding_emb_init)
    logger.info(F'Model Name: {model_name}')

    if mode is None or mode == 'train':
        logger.info('Loading Training and Validation Set')
        train_x, train_labels = get_data(data_cnf_train_texts, data_cnf_train_labels)

        if 'size' in data_cnf['valid']:
            # print(data_cnf['valid'])
            random_state = data_cnf['valid'].get('random_state', 1240)
            train_x, valid_x, train_labels, valid_labels = train_test_split(train_x, train_labels,
                                                                            test_size=data_cnf['valid']['size'],
                                                                            random_state=random_state)
        else:
            valid_x, valid_labels = get_data(data_cnf['valid']['texts'], data_cnf['valid']['labels'])
        mlb = get_mlb(data_cnf_labels_binarizer, np.hstack((train_labels, valid_labels)))
        train_y, valid_y = mlb.transform(train_labels), mlb.transform(valid_labels)
        labels_num = len(mlb.classes_)

        logger.info(F'Number of Labels: {labels_num}')
        logger.info(F'Size of Training Set: {len(train_x)}')
        logger.info(F'Size of Validation Set: {len(valid_x)}')

        logger.info('Training')
        train_dataset = MultiLabelDataset(train_x, train_y)
        valid_dataset = MultiLabelDataset(valid_x, valid_y, training=True)

        train_ds = ds.GeneratorDataset(train_dataset, column_names=["data", "label"], shuffle=True,
                                       num_parallel_workers=16,num_shards=device_num, shard_id=rank_id)
        train_ds = train_ds.batch(model_cnf['train']['batch_size'], drop_remainder=True,
                                  num_parallel_workers=16)
        #valid
        valid_ds = ds.GeneratorDataset(valid_dataset, ["data", "label"], shuffle=False,
                                       num_parallel_workers=16)
        valid_ds = valid_ds.batch(model_cnf['valid']['batch_size'], drop_remainder=False,
                                  num_parallel_workers=16)
        valid_ds_size = valid_ds.get_dataset_size()

        logger.info("labels_num:" + str(labels_num))
        logger.info(F"dataset size: {train_ds.get_dataset_size()}")
        network = model_dict[model_name](labels_num=labels_num, emb_init=emb_init,
                                         **data_cnf['model'], **model_cnf['model'])
        loss_fn = nn.BCEWithLogitsLoss()  # 定义损失函数
        net_with_loss = nn.WithLossCell(network, loss_fn)
        net_with_loss.set_train(True)
        valid_net = EvalNet(network, k=5)

        lr = nn.exponential_decay_lr(
            1e-3,
            0.9,
            int(train_ds.get_dataset_size() * model_cnf['train']['nb_epoch']),
            int(train_ds.get_dataset_size()),
            1,
            is_stair=False
        )

        optimizer = nn.Adam(params=network.trainable_params(), learning_rate=lr)
        train_net = XMLTrainOneStepCell(net_with_loss, optimizer=optimizer)
        save_steps = train_ds.get_dataset_size()

        time_cb = TimeMonitor()
        loss_cb = LossMonitor()
        cb = [time_cb, loss_cb]
        if rank_id == 0:
            config_ckp = CheckpointConfig(save_checkpoint_steps=save_steps, keep_checkpoint_max=20)
            ckpt_cb = ModelCheckpoint(prefix='corenet', directory=ckpt_save_dir,
                                      config=config_ckp)
            cb.append(ckpt_cb)
        if args.train_valid:
            eval_cb = EvalCallBack(valid_net, valid_ds, valid_ds_size, valid_y, 1, ckpt_save_dir, rank_id, tt=args.train_url)
            cb.append(eval_cb)
        model = Model(train_net)
        num_epochs = model_cnf['train']['nb_epoch']
        logger.info(F"epoch size: {num_epochs}")
        model.train(num_epochs, train_ds, callbacks=cb, dataset_sink_mode=True)
        if args.run_modelarts:
            try:
                mox.file.copy_parallel(args.train_url, obs_train_url)
                print("Successfully Upload {} to {}".format(args.train_url,
                                                            obs_train_url))
            except Exception as e:
                print('moxing upload {} to {} failed: '.format(args.train_url,
                                                               obs_train_url) + str(e))
        logger.info('Finish Training')
