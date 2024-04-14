from __future__ import annotations
from typing import Optional, Union


class Function():
    def __init__(self, *tensor:Tensor):
        parents = tensor

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError


    # input *tensor to compute return output with context created


# implement forward and backward pass
class Mul(Function):

    def forward(self, x, y):
        return x * y

    def backward(self, output):
        return output.gradient




class Tensor():
    def __init__(self, data:Union[int, float]):
        self.gradient:Optional[Tensor]
        self.data = data
        self._context:Optional[Function]

    # def topo_sort():

    # def backward():

    def mul(self, other):
        Mul.apply(self, other)

    def __mul__(self, other):
        self.mul(other)
