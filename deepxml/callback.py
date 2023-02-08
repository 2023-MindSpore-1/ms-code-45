import os
import numpy as np
import shutil
import mindspore as ms
from logzero import logger
from mindspore.train.callback import Callback
from mindspore.communication.management import get_rank
from mindspore.train.serialization import save_checkpoint
from deepxml.evaluation import get_p_5, get_n_5, get_p_1, get_n_1


class EvalCallBack(Callback):
    """
    Evaluation callback when training.

    Args:
        eval_function (function): evaluation function.
        eval_param_dict (dict): evaluation parameters' configure dict.
        interval (int): run evaluation interval, default is 1.
        eval_start_epoch (int): evaluation start epoch, default is 1.
        save_best_ckpt (bool): Whether to save best checkpoint, default is True.
        besk_ckpt_name (str): bast checkpoint name, default is `best.ckpt`.
        metrics_name (str): evaluation metrics name, default is `acc`.

    Returns:
        None

    Examples:
        >>> EvalCallBack(eval_function, eval_param_dict)
    """

    def __init__(self, net, dataset, dataset_size, valid_y, eval_step,
                 checkpoint_path=None, save_dir=None, rank_id=0, tt=0):
        super(EvalCallBack, self).__init__()
        self.tt = tt
        self.eval_step = eval_step
        self.dataset = dataset
        self.dataset_size = dataset_size
        self.checkpoint_path = checkpoint_path
        self.net = net
        self.valid_y = valid_y
        self.best_epoch = 0
        self.rank_id = rank_id
        self.save_dir = save_dir
        self.best_n5 = 0
        self.best_ckpt_path = os.path.abspath("./best_ckpt")
    # def step_end(self, run_context):
    #     cb_params = run_context.original_args()
    #     cur_step = cb_params.cur_step_num
    #     if cur_step % self.eval_step == 0:
    #         labels_all = []
    #         loss = 0
    #         for data in self.dataset.create_dict_iterator():
    #             valid_x = data['data']
    #             valid_y = data['label']
    #             valid_loss, labels = self.net(valid_x, valid_y)
    #             loss = valid_loss.asnumpy() + loss
    #             labels_all.append(labels.asnumpy())
    #         loss = loss / self.dataset_size
    #         labels_all = np.concatenate(labels_all)
    #         p5 = get_p_5(labels_all, self.valid_y)
    #         n5 = get_n_5(labels_all, self.valid_y)
    #         if n5 > self.best_n5:
    #             save_checkpoint(self.net, 'best.ckpt')
    #         logger.info(F'Valid loss: {loss}, P@5: {p5}, N@5: {n5}')

    def epoch_end(self, run_context):
        """Callback when epoch end."""
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        if cur_epoch % self.eval_step == 0:
            labels_all = []
            loss = 0
            for data in self.dataset.create_dict_iterator():
                valid_x = data['data']
                valid_y = data['label']
                valid_loss, labels = self.net(valid_x, valid_y)
                loss = valid_loss.asnumpy() + loss
                labels_all.append(labels.asnumpy())
            loss = loss / self.dataset_size
            labels_all = np.concatenate(labels_all)
            p5 = get_p_5(labels_all, self.valid_y)
            n5 = get_n_5(labels_all, self.valid_y)
            if n5 > self.best_n5:
                self.best_n5 = n5
                self.best_epoch = cur_epoch
                logger.info(F'Early Stop at : {self.best_epoch} epoch.')
                if self.rank_id == 0:
                    save_checkpoint(self.net, os.path.join(self.tt, 'best_%s_%s.ckpt' % (cur_epoch, n5)))
            p1 = get_p_1(labels_all, self.valid_y)
            n1 = get_n_1(labels_all, self.valid_y)
            logger.info(F'Valid loss: {loss}, P@1: {p1}, N@1: {n1}, P@5: {p5}, N@5: {n5}')