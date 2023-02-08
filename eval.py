import os
import click
import numpy as np
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

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=4)

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
class EvalNetWork(nn.Cell):
    def __init__(self, network, top_num):
        super(EvalNetWork, self).__init__()
        self.network = network
        self.topk = ops.TopK()
        self.sigmoid = ops.Sigmoid()
        self.k = top_num
    def construct(self,x):
        data_x = self.network(x)
        scores, labels = self.topk(data_x, self.k)
        data_pred = self.sigmoid(scores)
        return data_pred, labels

if __name__ == '__main__':
    yaml = YAML(typ='safe')

    data_cnf = 'configure/datasets/EUR-Lex.yaml'
    model_cnf = 'configure/models/CorNetXMLCNN-EUR-Lex.yaml'
    mode = None
    # profiles = Profiler()

    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
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
    network = model_dict[model_name](labels_num=labels_num, emb_init=emb_init,
                                     **data_cnf['model'],**model_cnf['model'])
    network.set_train(False)
    params = load_checkpoint('/mass_store/zjc/CorNet/CorNetXMLCNN-EUR-Lex.ckpt')
    load_param_into_net(network, params)
    logger.info('Load Checkpoint Success......')
    eval_net = EvalNetWork(network, top_num=100)
    score_list = []
    label_list = []
    logger.info('Start Evaluation......')
    for idx, data in enumerate(test_ds.create_dict_iterator()):
        logger.info(F'Evaluate Index {idx}')
        predict, labels = eval_net(data['data'])
        score_list.append(predict.asnumpy())
        label_list.append(labels.asnumpy())
    logger.info('Start Save Result.....')
    score_lists = np.concatenate(score_list)
    label_lists = np.concatenate(label_list)
    labels = mlb.classes_[label_lists]
    output_res(data_cnf['output']['res'], F'{model_name}-{data_name}', score_lists, labels)
    logger.info('Finish Training')
