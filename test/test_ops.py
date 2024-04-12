import unittest
import numpy as np
from fusion import Tensor
from tinygrad.tensor import Tensor as Tiny_Tensor


class test_gradient(unittest.TestCase):
    def test_specifique_values(self):
        x = Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2]])
        y = Tensor([[0, 1, 0, 4], [0, 0, 1, 7], [1, 1, 0, 8]])
        w = x * y
        v = w + y
        z = v.sum()
        z.backward()

        z_gradient = z.gradient.ndata
        assert z_gradient == 1
        assert z.shape == ()

        v_gradient = v.gradient.ndata
        assert v_gradient.tolist() == [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        assert v_gradient.shape == (3, 4)

        w_gradient = w.gradient.ndata
        assert w_gradient.tolist() == [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        assert w_gradient.shape == (3, 4)

        x_gradient = x.gradient.ndata
        assert x_gradient.tolist() == [[0, 1, 0, 4], [0, 0, 1, 7], [1, 1, 0, 8]]
        assert x_gradient.shape == (3, 4)

        y_gradient = y.gradient.ndata
        assert y_gradient.tolist() == [[2, 3, 4, 5], [6, 7, 8, 9], [10, 1, 2, 3]]
        assert y_gradient.shape == (3, 4)

    def test_compare_tinygrad_on_random_int(self):

        x_val = np.random.randint(9, size=(3,3)).tolist()
        y_val = np.random.randint(9, size=(3,3)).tolist()

        x = Tensor(x_val)
        y = Tensor(y_val)
        w = x * y
        v = w + y
        z = v.sum()
        z.backward()
        z_gradient = z.gradient.ndata
        v_gradient = v.gradient.ndata
        w_gradient = w.gradient.ndata
        x_gradient = x.gradient.ndata
        y_gradient = y.gradient.ndata

        x_tiny = Tiny_Tensor(x_val, requires_grad=True)
        y_tiny = Tiny_Tensor(y_val, requires_grad=True)
        w_tiny = x_tiny * y_tiny
        v_tiny = w_tiny + y_tiny
        z_tiny = v_tiny.sum()
        z_tiny.backward()
        z_tiny_grad = z_tiny.grad.numpy()
        v_tiny_grad = v_tiny.grad.numpy()
        w_tiny_grad = w_tiny.grad.numpy()
        x_tiny_grad = x_tiny.grad.numpy()
        y_tiny_grad = y_tiny.grad.numpy()

        assert z.ndata == z_tiny.numpy()
        assert x_gradient.tolist() == x_tiny_grad.tolist()
        assert y_gradient.tolist() == y_tiny_grad.tolist()
        assert w_gradient.tolist() == w_tiny_grad.tolist()
        assert v_gradient.tolist() == v_tiny_grad.tolist()
        assert z_gradient.tolist() == z_tiny_grad.tolist()

        # print("\n")
        # print(f"{x_gradient.tolist()} == {x_tiny_grad.tolist()}")
        # print(f"{y_gradient.tolist()} == {y_tiny_grad.tolist()}")
        # print(f"{w_gradient.tolist()} == {w_tiny_grad.tolist()}")
        # print(f"{v_gradient.tolist()} == {v_tiny_grad.tolist()}")
        # print(f"{z_gradient.tolist()} == {z_tiny_grad.tolist()}")


if __name__ == "__main__":
    unittest.main()
