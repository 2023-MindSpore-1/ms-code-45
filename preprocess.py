import os
import click
import numpy as np
import argparse
from pathlib import Path
from ruamel.yaml import YAML
from sklearn.model_selection import train_test_split
from logzero import logger
from mindspore import nn
from mindspore import Model
from mindspore import ops
from mindspore.profiler import Profiler
from mindspore.train.callback import ModelCheckpoint, LossMonitor, TimeMonitor, CheckpointConfig
from mindspore.train import load_checkpoint, load_param_into_net
from deepxml.dataset import MultiLabelDataset
from deepxml.data_utils import get_data, get_mlb, get_word_emb, output_res

from deepxml.xmlcnn import XMLCNN, CorNetXMLCNN
from deepxml.trainonestep import XMLTrainOneStepCell
from deepxml.models import CoreModel
import mindspore.dataset as ds
from mindspore import context
import mindspore

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=0)
parser = argparse.ArgumentParser(description='Train CorNet')
parser.add_argument('--dataset_path',
                    type=str,
                    default="/mnt/nvme1/deep_data",
                    help='path where the dataset is saved')
parser.add_argument('--output_path',
                    type=str,
                    default="./preprocess_Result",
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
if __name__ == '__main__':
    yaml = YAML(typ='safe')

    data_cnf = '../configure/datasets/EUR-Lex.yaml'
    model_cnf = '../configure/models/CorNetXMLCNN-EUR-Lex.yaml'
    mode = None
    # profiles = Profiler()

    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
    data_cnf['embedding']['emb_init'] = os.path.join(args.dataset_path, 'emb_init.npy')
    data_cnf['labels_binarizer'] = os.path.join(args.dataset_path, 'labels_binarizer')
    data_cnf['test']['texts'] = os.path.join(args.dataset_path, 'test_texts.npy')
    model, model_name, data_name = None, model_cnf['name'], data_cnf['name']
    model_path = os.path.join(model_cnf['path'], F'{model_name}-{data_name}')
    emb_init = get_word_emb(data_cnf['embedding']['emb_init'])
    logger.info(F'Model Name: {model_name}')

    logger.info('Loading Training and Validation Set')
    test_x, _ = get_data(data_cnf['test']['texts'], None)
    logger.info(F'Size of Test Set: {len(test_x)}')
    mlb = get_mlb(data_cnf['labels_binarizer'])
    labels_num = len(mlb.classes_)
    test_dataset = MultiLabelDataset(test_x, training=False)

    test_ds = ds.GeneratorDataset(test_dataset, column_names=["data"], shuffle=False,
                                   num_parallel_workers=16)
    test_ds = test_ds.batch(1, drop_remainder=True,
                              num_parallel_workers=16)

    logger.info("labels_num:" + str(labels_num))
    logger.info(F"dataset size: {test_ds.get_dataset_size()}")
    img_path = os.path.join(args.output_path, 'img_data')
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    logger.info('Start Preprocess......')
    for idx, data in enumerate(test_ds.create_dict_iterator(output_numpy=True,num_epochs=1)):
        # logger.info(F'Evaluate Index {idx}')
        img_data = data['data'].astype(np.int32)
        file_name = "text_{}.bin".format(str(idx))
        img_file_path = os.path.join(img_path, file_name)
        img_data.tofile(img_file_path)
    #     predict, labels = eval_net(data['data'])
    #     score_list.append(predict.asnumpy())
    #     label_list.append(labels.asnumpy())
    # logger.info('Start Save Result.....')
    # score_lists = np.concatenate(score_list)
    # label_lists = np.concatenate(label_list)
    # labels = mlb.classes_[label_lists]
    # output_res(data_cnf['output']['res'], F'{model_name}-{data_name}', score_lists, labels)
    logger.info('Finish Preprocess')
