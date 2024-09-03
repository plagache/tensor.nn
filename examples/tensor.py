from __future__ import annotations

import unittest
from typing import Optional, Type

import numpy as np
from tinygrad import Tensor as Tiny_Tensor


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
