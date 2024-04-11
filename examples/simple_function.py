from __future__ import annotations
from typing import Optional, Union


class Function():
    def __init__(self, *tensor:Tensor):
        parents = tensor

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError


    @classmethod
    def apply(cls:type[Function], self, parents:Tensor):
        context = cls
        self._context.parents = parents
        self._context = context
        return self


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
        self._context:Optional[None]

    # def topo_sort():

    # def backward():

    def mul(self, other):
        Mul.apply(self, other)

    def __mul__(self, other):
        self.mul(other)
