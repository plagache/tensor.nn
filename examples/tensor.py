from __future__ import annotations

import unittest
from typing import Optional, Type

import numpy as np
from tinygrad import Tensor as Tiny_Tensor


class Function:
    def __init__(self, *tensor: Tensor):
        self.parents = tensor

    def forward(self, *args):
        return NotImplementedError

    def backward(self, *args):
        return NotImplementedError

    @classmethod
    def apply(cls, *tensor):
        context = cls(*tensor)
        output = Tensor(context.forward(*tensor))
        output._context = context
        return output


class Sum(Function):
    def forward(self, x: Tensor):
        return np.sum(x.data)

    def backward(self, output: Tensor):
        (x,) = self.parents
        return np.ones_like(x.data) * output.data


class Mul(Function):
    def forward(self, x: Tensor, y:Tensor):
        return np.multiply(x.data, y.data)

    def backward(self, output):
        (x, y) = self.parents
        return y.data * output.data, x.data * output.data


class Add(Function):
    def forward(self, x: Tensor, y: Tensor):
        return np.add(x.data, y.data)

    def backward(self, output: Tensor):
        (x, y) = self.parents
        return y.data, x.data


class Tensor:
    def __init__(self, data):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)
        self.gradient = None
        self._context: Optional[Function]

    def topological_sort(self):
        def _topological_sort(node, unique, sorted):
            print(node.data)
            print(node._context.parents)
            if self._context is not None:
                unique.add(node)
                if node not in sorted:
                    sorted.append(node)
                    _topological_sort(node, unique, sorted)
            return sorted

        return _topological_sort(self, set(), [])

    def backward(self):
        for node in reversed(self.topological_sort()):
            print(f"node: {node}\n")
        return

    def mul(self, x):
        return Mul.apply(self, x)

    def __mul__(self, x):
        return self.mul(x)

    def add(self, x):
        return Add.apply(self, x)

    def __add__(self, x):
        return self.add(x)

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
        sum.backward()

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
        tiny_sum.backward()

        # tiny_sum_gradient = tiny_sum.grad.numpy()
        # tiny_add_gradient = tiny_add.grad.numpy()
        # tiny_mul_gradient = tiny_mul.grad.numpy()
        # tiny_x_gradient = tiny_x.grad.numpy()
        # tiny_y_gradient = tiny_y.grad.numpy()

        # print(mul.data, tiny_mul.numpy())
        # print(add.data, tiny_add.numpy())
        print(sum.data, tiny_sum.numpy())
        # print(type(sum.data), type(tiny_sum.numpy()))
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
