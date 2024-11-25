from __future__ import annotations

import unittest
from typing import Optional

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
    def forward(self, x: Tensor, y: Tensor):
        return np.multiply(x.data, y.data)

    def backward(self, output: Tensor):
        (x, y) = self.parents
        return y.data * output.data, x.data * output.data


class Add(Function):
    def forward(self, x: Tensor, y: Tensor):
        return np.add(x.data, y.data)

    def backward(self, output: Tensor):
        return output.data, output.data


class Tensor:
    def __init__(self, data):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)
        self.gradient: Optional[Tensor] = None
        self._context: Optional[Function] = None

    def __repr__(self):
        return f"{self.data} , {self.gradient}"

    def topological_sort(self):
        """
        Produce the topological sort of the computation graph
        """

        def _topological_sort(node, unique, sorted):
            if getattr(node, "_context", None):
                for parent in node._context.parents:
                    if parent not in unique:
                        unique.add(parent)
                    _topological_sort(parent, unique, sorted)
                sorted.append(node)
            return sorted

        return _topological_sort(self, set(), [])

    def backward(self):
        """
        Propagate gradient for each Tensor backward in the computation graph
        """
        self.gradient = Tensor(1)

        for node in reversed(self.topological_sort()):
            if hasattr(node, "_context"):
                parents = node._context.parents
                gradients = node._context.backward(node.gradient)
                if len(parents) == 1:
                    gradients = [Tensor(gradients)]
                else:
                    gradients = [Tensor(gradient) for gradient in gradients]
                for parent, gradient in zip(parents, gradients):
                    parent.gradient = gradient if parent.gradient is None else parent.gradient + gradient
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
        t_sum = add.sum()
        t_sum.backward()

        sum_gradient = t_sum.gradient.data
        add_gradient = add.gradient.data
        mul_gradient = mul.gradient.data
        x_gradient = xt.gradient.data
        y_gradient = yt.gradient.data

        tiny_x = Tiny_Tensor(xs, requires_grad=True)
        tiny_y = Tiny_Tensor(ys, requires_grad=True)

        tiny_mul = tiny_x * tiny_y
        tiny_add = tiny_mul + tiny_x
        tiny_sum = tiny_add.sum()
        tiny_sum.backward()

        tiny_sum_gradient = tiny_sum.grad.numpy()
        tiny_add_gradient = tiny_add.grad.numpy()
        tiny_mul_gradient = tiny_mul.grad.numpy()
        tiny_x_gradient = tiny_x.grad.numpy()
        tiny_y_gradient = tiny_y.grad.numpy()

        # print("\nx:", xt.data)
        # print("\ny:", yt.data)
        # print("\nmul result:", mul.data, tiny_mul.numpy())
        # print("\nadd result:", add.data, tiny_add.numpy())
        # print("\nsum result:", t_sum.data, tiny_sum.numpy())
        np.testing.assert_allclose(t_sum.data, tiny_sum.numpy())
        np.testing.assert_allclose(add.data, tiny_add.numpy())
        np.testing.assert_allclose(mul.data, tiny_mul.numpy())

        # print("compare grad sum: ", sum_gradient, "\n", tiny_sum_gradient)
        # print("compare grad add: ", add_gradient, "\n", tiny_add_gradient)
        # print("compare grad mul: ", mul_gradient, "\n", tiny_mul_gradient)
        # print("compare grad x: ", x_gradient, "\n", tiny_x_gradient)
        # print("compare grad y: ", y_gradient, "\n", tiny_y_gradient)
        np.testing.assert_allclose(sum_gradient, tiny_sum_gradient)
        np.testing.assert_allclose(add_gradient, tiny_add_gradient)
        np.testing.assert_allclose(mul_gradient, tiny_mul_gradient)
        np.testing.assert_allclose(x_gradient, tiny_x_gradient)
        np.testing.assert_allclose(y_gradient, tiny_y_gradient)


if __name__ == "__main__":
    unittest.main()
