from __future__ import annotations

from typing import Optional, Union, Type


class Function():
    def __init__(self, *tensor:Tensor):
        # if need_gradient add parents to construct graph
        self.parents = tensor

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

    # Classmethod takes as first argument the class or subclass itself
    # input *parents_tensor to compute return new tensor with context
    # the goal is to use the correct subclass of the Function and being able to use the corresponding backward
    @classmethod
    def apply(cls: Type[Function], *parents):
        print(f"cls: {cls}")
        context = cls(*parents)
        print("context: ", context)
        output = Tensor(context.forward(*parents))
        # output = Tensor(cls(*parents).forward(*parents))
        print("context forward: ", context.forward(*parents))
        output._context = context
        # print("tensor: ", output)
        # print("tensor data: ", output.data)
        # print("tensor context: ", output._context)
        # print("tensor context parents: ", output._context.parents)
        print("Return -----------------------")
        return output


class Mul(Function):
    def forward(self, x: Tensor, y: Tensor):
        return x.data * y.data

    def backward(self, output: Tensor):
        return output.data * self.parents[1].data, output.data * self.parents[0].data


class Add(Function):
    def forward(self, x: Tensor, y: Tensor):
        return x.data + y.data

    def backward(self, output: Tensor):
        return output.data, output.data


class Tensor():
    def __init__(self, data:Union[int, float]):
        self.gradient:Optional[Tensor] = None
        self.data = data
        self._context:Optional[Function] = None

    # def topo_sort():

    # def backward(self):
    #     print(self._context)
    #     if self._context is not None:
    #         for parent in self._context.parents:
    #             self.backward()

    def mul(self, other):
        return Mul.apply(self, other)

    def __mul__(self, other):
        return self.mul(other)

    def add(self, other):
        return Add.apply(self, other)

    def __add__(self, other):
        return self.add(other)

if __name__ == "__main__":
    x = Tensor(3)
    y = Tensor(8)
    z = x + y
    b = z * x
    r = z + y
    d = r * x
