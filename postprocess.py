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
from sklearn.preprocessing import MultiLabelBinarizer
from deepxml.evaluation import get_p_1, get_p_3, get_p_5, get_n_1, get_n_3, get_n_5, get_inv_propensity
from deepxml.evaluation import get_psp_1, get_psp_3, get_psp_5, get_psndcg_1, get_psndcg_3, get_psndcg_5
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
parser.add_argument('--result_path',
                    type=str,
                    default="./result_Files",
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
    data_cnf_test_labels = os.path.join(args.dataset_path, 'test_labels.npy')
    data_cnf_train_labels = os.path.join(args.dataset_path, 'train_labels.npy')
    data_cnf['output']['res'] = './results'
    model, model_name, data_name = None, model_cnf['name'], data_cnf['name']
    model_path = os.path.join(model_cnf['path'], F'{model_name}-{data_name}')
    emb_init = get_word_emb(data_cnf['embedding']['emb_init'])
    logger.info(F'Model Name: {model_name}')

    logger.info('Loading Training and Validation Set')
    test_x, _ = get_data(data_cnf['test']['texts'], None)
    logger.info(F'Size of Test Set: {len(test_x)}')
    mlb = get_mlb(data_cnf['labels_binarizer'])
    labels_num = len(mlb.classes_)
    logger.info(F'Size of labels_num: {labels_num}')
    test_dataset = MultiLabelDataset(test_x, training=False)

    test_ds = ds.GeneratorDataset(test_dataset, column_names=["data"], shuffle=False,
                                   num_parallel_workers=16)
    test_ds = test_ds.batch(1, drop_remainder=True,
                              num_parallel_workers=16)

    logger.info("labels_num:" + str(labels_num))
    logger.info(F"dataset size: {test_ds.get_dataset_size()}")
    img_path = os.path.join(args.result_path)
    score_list = []
    label_list = []

    logger.info('Start Preprocess......')
    for idx, data in enumerate(test_ds.create_dict_iterator(output_numpy=True,num_epochs=1)):
        predict_name = "text_{}_0.bin".format(str(idx))
        labels_name = "text_{}_1.bin".format(str(idx))
        predict = os.path.join(img_path, predict_name)
        labels = os.path.join(img_path, labels_name)
        predict = np.fromfile(predict,dtype=np.float32).reshape(1, 100)
        labels = np.fromfile(labels, dtype=np.int32).reshape(1, 100)
        score_list.append(predict)
        label_list.append(labels)
    logger.info('Start Save Result.....')
    score_lists = np.concatenate(score_list)
    label_lists = np.concatenate(label_list)
    labels = mlb.classes_[label_lists]
    res = labels
    a = 0.55
    b = 1.5
    targets = np.load(data_cnf_test_labels, allow_pickle=True)
    train_labels = np.load(data_cnf_train_labels, allow_pickle=True)
    evalmlb = MultiLabelBinarizer(sparse_output=True)
    targets = evalmlb.fit_transform(targets)
    inv_w = get_inv_propensity(evalmlb.transform(train_labels), a, b)
    p1 = get_p_1(res, targets, evalmlb)
    p3 = get_p_3(res, targets, evalmlb)
    p5 = get_p_5(res, targets, evalmlb)
    logger.info(F'Precision@1: {p1}, P@3: {p3}, P@5: {p5}')
    ndcg1 = get_n_1(res, targets, evalmlb)
    ndcg3 = get_n_3(res, targets, evalmlb)
    ndcg5 = get_n_5(res, targets, evalmlb)
    logger.info(F'nDCG@1: {ndcg1}, nDCG@3: {ndcg3}, nDCG@5: {ndcg5}')
    psp1 = get_psp_1(res, targets, inv_w, evalmlb)
    psp3 = get_psp_3(res, targets, inv_w, evalmlb)
    psp5 = get_psp_5(res, targets, inv_w, evalmlb)
    logger.info(F'PSPrecision@1: {psp1}, PSPrecision@3: {psp3}, PSPrecision@5: {psp5}')
    psndcg1 = get_psndcg_1(res, targets, inv_w, evalmlb)
    psndcg3 = get_psndcg_3(res, targets, inv_w, evalmlb)
    psndcg5 = get_psndcg_5(res, targets, inv_w, evalmlb)
    logger.info(F'PSPnDCG@1: {psndcg1}, PSPnDCG@3: {psndcg3}, PSPnDCG@5: {psndcg5}')
    output_res(data_cnf['output']['res'], F'{model_name}-{data_name}', score_lists, labels)
    logger.info('Finish Acc')
