import numpy as np
from deepxml.cornet import CorNet
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.initializer as init

from mindspore import Parameter, Tensor, ms_function
from mindspore import Tensor
from mindspore.common import dtype as mstype
import mindspore.numpy as mnp
import mindspore
from mindspore import context
# context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=3)

class XMLCNN(nn.Cell):
    def __init__(self, dropout, labels_num,  bottleneck_dim, num_filters, dynamic_pool_length,
                 vocab_size=None, emb_size=None, emb_trainable=True, emb_init=None, padding_idx=0, **kwargs):
        super(XMLCNN, self).__init__()
        emb_init = np.array(emb_init).astype(np.float32)
        if emb_init is not None:
            if vocab_size is not None:
                assert vocab_size == emb_init.shape[0]
            if emb_size is not None:
                assert emb_size == emb_init.shape[1]
            vocab_size, emb_size = emb_init.shape
        self.emb_init = Tensor(emb_init)

        self.output_channel = num_filters
        self.num_bottleneck_hidden = bottleneck_dim
        self.dynamic_pool_length = dynamic_pool_length

        #         self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx, sparse=True,
        #                                 _weight=torch.from_numpy(emb_init).float() if emb_init is not None else None)
        self.emb = nn.Embedding(vocab_size, emb_size,
                                use_one_hot=True,
                                embedding_table=self.emb_init,
                                padding_idx=padding_idx).to_float(mstype.float16)
        ##self.emb.weight.requires_grad = emb_trainable  ##mindspore的Embedding没有weight这个参数

        self.ks = 3  # There are three conv nets here
        ## Different filter sizes in xml_cnn than kim_cnn
        self.relu = ops.ReLU()
        self.conv1 = nn.Conv1d(300, self.output_channel, 2,
                               pad_mode="pad", padding=1,
                               weight_init='XavierUniform')
        # self.conv1.conv2d.add_prim_attr("primitive_target", "CPU")
        self.conv2 = nn.Conv1d(300, self.output_channel, 4,
                               pad_mode="pad", padding=3,
                               weight_init='XavierUniform')
        # self.conv2.conv2d.add_prim_attr("primitive_target", "CPU")
        self.conv3 = nn.Conv1d(300, self.output_channel, 8,
                               pad_mode="pad", padding=7,
                               weight_init='XavierUniform')
        # self.conv3.conv2d.add_prim_attr("primitive_target", "CPU")
        ##self.pool = nn.AdaptiveMaxPool1d(self.dynamic_pool_length)  # Adaptive pooling 8 输出的最后一维是8
        # self.pool = ops.AdaptiveAvgPool2D(self.dynamic_pool_length)
        self.pool1 = nn.MaxPool1d(kernel_size=67, stride=62)  # default-pad_mode='valid'
        self.pool2 = nn.MaxPool1d(kernel_size=69, stride=62)
        self.pool3 = nn.MaxPool1d(kernel_size=66, stride=63)

        self.bottleneck = nn.Dense(self.ks * self.output_channel * self.dynamic_pool_length,
                                   self.num_bottleneck_hidden,weight_init='XavierUniform')#.to_float(mstype.float16)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Dense(self.num_bottleneck_hidden, labels_num, weight_init='XavierUniform')#.to_float(mstype.float16)
        self.expand_dims = ops.ExpandDims()
        self.concat_op = ops.Concat(axis=1)
        self.cast = ops.Cast()

    def construct(self, x):
        # x = self.cast(x, mstype.int32)
        embe_out = self.emb(x)  # (batch, sent_len, embed_dim)
        embe_out = self.cast(embe_out, mstype.float32)
        # x0 = self.expand_dims(embe_out, 1)

        # embe_out: (32, 500, 300)
        x = ops.transpose(embe_out, (0, 2, 1))
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x3 = self.relu(self.conv3(x))

        x1 = self.pool1(x1)
        x2 = self.pool2(x2)
        x3 = self.pool3(x3)

        x = self.concat_op((x1, x2, x3))
        x = x.view(-1, self.ks * self.output_channel * self.dynamic_pool_length)
        # x = self.relu(self.bottleneck(self.cast(x, mstype.float16)))
        x = self.relu(self.bottleneck(x))
        # x = self.cast(x, mstype.float32)
        x = self.dropout(x)
        # x = self.cast(x,mstype.float16)
        logit = self.fc1(x)  # (batch, target_size)
        # logit = self.cast(logit, mstype.float32)
        return logit


class CorNetXMLCNN(nn.Cell):
    def __init__(self, dropout, labels_num, dynamic_pool_length,
                 bottleneck_dim, num_filters, **kwargs):
        super(CorNetXMLCNN, self).__init__()
        # self.xmlcnn = XMLCNN(dropout, labels_num, dynamic_pool_length, bottleneck_dim, num_filters, **kwargs)
        self.xmlcnn = XMLCNN(dropout=dropout, labels_num=labels_num,
                             dynamic_pool_length=dynamic_pool_length,
                             bottleneck_dim=bottleneck_dim,
                             num_filters=num_filters, **kwargs)
        self.cornet = CorNet(labels_num, **kwargs)

    def construct(self, input_variables):
        raw_logits = self.xmlcnn(input_variables)
        cor_logits = self.cornet(raw_logits)
        return cor_logits

if __name__ == '__main__':

    import random
    import numpy as np

    # input = np.random.rand(6, 3, 32, 32)
    inputs = np.random.randint(low=1, high=10000, size=(40, 500)).astype(np.int32)
    print(inputs.dtype)
    print(inputs.shape)
    inputs = Tensor(inputs)

    # network = XMLCNN(dropout=0.5, labels_num=3801,dynamic_pool_length=8, bottleneck_dim=512, num_filters=128, vocab_size=500, embedding_size=300)
    network = CorNetXMLCNN(dropout=0.5, labels_num=3801,dynamic_pool_length=8, bottleneck_dim=512, num_filters=128, vocab_size=500, embedding_size=300)
    logits = network(inputs)
    print(logits.asnumpy())
    # print(logits.shape)