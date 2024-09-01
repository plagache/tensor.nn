from __future__ import annotations

import unittest
from typing import Optional

import numpy as np
from tinygrad import Tensor as Tiny_Tensor


class Function:
    def __init__(self, *tensor: Tensor):
        self.parents = tensor

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

    @classmethod
    def apply(cls: type[Function], *parents):
        context = cls(*parents)
        output = Tensor(context.forward(*parents))
        output._context = context
        return output


class Sum(Function):
    def forward(self, x: Tensor):
        return np.sum(x.data)

    def backward(self, output: Tensor):
        (x,) = self.parents
        return np.ones_like(x.data) * 1


class Add(Function):
    def forward(self, x: Tensor, y: Tensor):
        return np.add(x.data, y.data)

    def backward(self, output: Tensor):
        return output.gradient


class Mul(Function):
    def forward(self, x: Tensor, y: Tensor):
        return np.multiply(x.data, y.data)

    def backward(self, output: Tensor):
        x, y = self.parents
        return np.multiply(y.data, output.gradient), np.multiply(x.data, output.gradient)


class Tensor:
    def __init__(self, data):
        self.gradient: Optional[np.ndarray]
        self._context: Optional[Function]
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)

    def topo_sort():
        return

    def backward():
        return

    def add(self, other):
        return Add.apply(self, other)

    def __add__(self, other):
        return self.add(other)

    def mul(self, other: Tensor):
        return Mul.apply(self, other)

    def __mul__(self, other):
        return self.mul(other)

    def sum(self):
        return Sum.apply(self)


class test_tensor(unittest.TestCase):
    def test_tensor_with_tinygrad(self):
        xs = np.random.randint(-9, 9, size=(3, 3))
        ys = np.random.randint(-9, 9, size=(3, 3))

        xt = Tensor(xs)
        yt = Tensor(ys)

        mul = xt * yt
        add = mul + xt
        sum = add.sum()
        # sum.backward()

        # sum_gradient = sum.gradient
        # add_gradient = add.gradient
        # mul_gradient = mul.gradient
        # x_gradient = xt.gradient
        # y_gradient = yt.gradient

        tiny_x = Tiny_Tensor(xs, requires_grad=True)
        tiny_y = Tiny_Tensor(ys, requires_grad=True)

        tiny_mul = tiny_x * tiny_y
        tiny_add = tiny_mul + tiny_x
        tiny_sum = tiny_add.sum()
        # tiny_sum.backward()

        # tiny_sum_gradient = tiny_sum.grad.numpy()
        # tiny_add_gradient = tiny_add.grad.numpy()
        # tiny_mul_gradient = tiny_mul.grad.numpy()
        # tiny_x_gradient = tiny_x.grad.numpy()
        # tiny_y_gradient = tiny_y.grad.numpy()

        print(mul.data, tiny_mul.numpy())
        print(sum.data, tiny_sum.numpy())
        print(type(sum.data), tiny_sum.numpy())
        np.testing.assert_allclose(sum.data, tiny_sum.numpy())
        np.testing.assert_allclose(add.data, tiny_add.numpy())
        np.testing.assert_allclose(mul.data, tiny_mul.numpy())

        # np.testing.assert_allclose(sum_gradient, tiny_sum_gradient)
        # np.testing.assert_allclose(add_gradient, tiny_add_gradient)
        # np.testing.assert_allclose(mul_gradient, tiny_mul_gradient)
        # np.testing.assert_allclose(x_gradient, tiny_x_gradient)
        # np.testing.assert_allclose(y_gradient, tiny_y_gradient)


if __name__ == "__main__":
    unittest.main()
