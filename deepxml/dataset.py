import numpy as np
import mindspore.dataset as ds
# from scipy.sparse import csr_matrix
# from typing import Sequence, Optional

# TDataX = Sequence[Sequence]
# TDataY = Optional[csr_matrix]


class MultiLabelDataset():
    def __init__(self, data_x=0, data_y=0, training=True):
        self.data_x, self.data_y, self.training = data_x, data_y, training
        # self.data_x = np.random.randint(low=1, high=10000, size= (4, 50) )
    def __getitem__(self, item):
        data_x = self.data_x[item]
        # print('item:',data_x.shape)
        # return data_x.astype('float32')

        if self.training and self.data_y is not None:
            data_y = self.data_y[item].toarray().squeeze(0).astype(np.float32)
            return data_x, data_y
        else:
            return data_x

    def __len__(self):
        return len(self.data_x)




