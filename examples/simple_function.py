from __future__ import annotations
from typing import Optional, Union


class Function():
    def __init__(self, *tensor:Tensor):
        # if need_gradient add parents
        self.parents = tensor

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError


    # input *tensor to compute return output tensor with context created
    # the goal is to use the correct subclass of the Function and being able to use the correct backward
    @classmethod
    def apply(cls:type[Function], *forward_input):
        context = cls(*forward_input)
        output = Tensor(context.forward(*forward_input))
        output._context = context
        return output



# implement forward and backward pass
class Mul(Function):

    def forward(self, x, y):
        return x * y

    def backward(self, output):
        return output.gradient + self


class Add(Function):

    def forward(self, x, y):
        return x + y

    def backward(self, output):
        return output.gradient


class Tensor():
    def __init__(self, data:Union[int, float]):
        self.gradient:Optional[Tensor]
        self.data = data
        self._context:Optional[Function]

    # def topo_sort():

    def backward(self):
        print(self._context)
        if self._context is not None:
            for parent in self._context.parents:
                self.backward()


    def mul(self, other):
        Mul.apply(self, other)

    def __mul__(self, other):
        self.mul(other)

    def add(self, other):
        Add.apply(self, other)

    def __add__(self, other):
        self.add(other)
