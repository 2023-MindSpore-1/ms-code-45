from collections import deque
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.parameter import ParameterTuple
from mindspore.context import ParallelMode
from mindspore import nn
from mindspore.communication.management import get_group_size
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore import context
from mindspore import ops
from mindspore import numpy as msnp

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0

clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad

class XMLTrainOneStepCell(nn.Cell):
    """
    Encapsulation class of fasttext network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """
    def __init__(self, network, optimizer, sens=1.0):
        super(XMLTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode not in ParallelMode.MODE_LIST:
            raise ValueError("Parallel mode does not support: ", self.parallel_mode)
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = None
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)

        self.hyper_map = C.HyperMap()
        self.cast = P.Cast()
        self.gradient_norm_queue = deque([msnp.inf], maxlen=5)
        self.norm = nn.Norm()
        self.mul = ops.Mul()
        self.norm_type = 2.0

    def set_sens(self, value):
        self.sens = value

    def construct(self, x, y):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(x, y)
        grads = self.grad(self.network, weights)(x, y,
                                                 self.cast(F.tuple_to_array((self.sens,)),
                                                           mstype.float32))
        # grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        _, grads = self.clip_grad_norm_(grads, 1.0)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))

    def clip_grad_norm_(self, parameters, max_norm):
        max_norm = max_norm
        norm_type = self.norm_type
        total_norm = 0
        for p in parameters:
            param_norm = self.norm(p)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            new_grads = ()
            for p in parameters:
                new_grads += (self.mul(p, clip_coef),)
            parameters = new_grads
        return total_norm, parameters

    def clip_gradient(self, grads):
        if self.gradient_clip_value is not None:
            print(self.gradient_norm_queue)
            max_norm = msnp.max(self.gradient_norm_queue)
            total_norm, grads = self.clip_grad_norm_(grads, max_norm * self.gradient_clip_value)
            self.gradient_norm_queue.append(msnp.min(total_norm, max_norm * 2.0, 1.0))
        return grads
