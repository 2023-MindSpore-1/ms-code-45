import os
import click
import numpy as np
import argparse
from pathlib import Path
from ruamel.yaml import YAML
from logzero import logger
import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore.train import load_checkpoint, load_param_into_net
from deepxml.data_utils import get_data, get_mlb, get_word_emb, output_res

from deepxml.xmlcnn import XMLCNN, CorNetXMLCNN
from deepxml.models import CoreModel
from mindspore import context

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=0)
parser = argparse.ArgumentParser(description='Train CorNet')
parser.add_argument('--dataset_path',
                    type=str,
                    default="/mnt/nvme1/deep_data",
                    help='path where the dataset is saved')
parser.add_argument('--checkpoint_path',
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
    data_cnf['embedding']['emb_init'] = os.path.join(args.dataset_path, 'emb_init.npy')
    data_cnf['labels_binarizer'] = os.path.join(args.dataset_path, 'labels_binarizer')
    model, model_name, data_name = None, model_cnf['name'], data_cnf['name']
    model_path = os.path.join(model_cnf['path'], F'{model_name}-{data_name}')
    emb_init = get_word_emb(data_cnf['embedding']['emb_init'])
    logger.info(F'Model Name: {model_name}')

    mlb = get_mlb(data_cnf['labels_binarizer'])
    labels_num = len(mlb.classes_)
    network = model_dict[model_name](labels_num=labels_num, emb_init=emb_init,
                                     **data_cnf['model'],**model_cnf['model'])
    network.set_train(False)
    params = load_checkpoint(args.checkpoint_path)
    load_param_into_net(network, params)
    logger.info('Load Checkpoint Success......')
    eval_net = EvalNetWork(network, top_num=100)
    inputs = ms.numpy.zeros([1, 500], ms.int32)
    ms.export(eval_net, inputs, file_name='CoreNetXML', file_format='MINDIR')
