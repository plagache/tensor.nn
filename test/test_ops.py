import unittest
import numpy as np
from fusion import Tensor
from tinygrad.tensor import Tensor as Tiny_Tensor
# from tinygrad.dtype import dtypes


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
        x_val = np.random.randint(9, size=(3, 3)).tolist()
        y_val = np.random.randint(9, size=(3, 3)).tolist()
        # x_val = np.random.uniform(-9, 9, size=(3, 3)).tolist()
        # y_val = np.random.uniform(-9, 9, size=(3, 3)).tolist()
        # print(x_val)
        # print(y_val)

        x = Tensor(x_val)
        y = Tensor(y_val)
        w = x * y
        v = w + y
        r = v.relu()
        z = r.sum()
        z.backward()
        z_gradient = z.gradient.ndata
        r_gradient = r.gradient.ndata
        v_gradient = v.gradient.ndata
        w_gradient = w.gradient.ndata
        x_gradient = x.gradient.ndata
        y_gradient = y.gradient.ndata

        # x_tiny = Tiny_Tensor(x_val, requires_grad=True, dtype=dtypes.int)
        # y_tiny = Tiny_Tensor(y_val, requires_grad=True, dtype=dtypes.int)
        x_tiny = Tiny_Tensor(x_val, requires_grad=True)
        y_tiny = Tiny_Tensor(y_val, requires_grad=True)
        w_tiny = x_tiny * y_tiny
        v_tiny = w_tiny + y_tiny
        r_tiny = v_tiny.relu()
        z_tiny = r_tiny.sum()
        z_tiny.backward()
        z_tiny_grad = z_tiny.grad.numpy()
        r_tiny_grad = r_tiny.grad.numpy()
        v_tiny_grad = v_tiny.grad.numpy()
        w_tiny_grad = w_tiny.grad.numpy()
        x_tiny_grad = x_tiny.grad.numpy()
        y_tiny_grad = y_tiny.grad.numpy()

        # print(z.ndata, z_tiny.numpy())
        # print(z_tiny.numpy())
        # print(z_tiny.numpy().data.tobytes())
        # print(x_tiny.numpy().data.tobytes())
        assert z.ndata == z_tiny.numpy()
        assert z_gradient.tolist() == z_tiny_grad.tolist()
        assert z_gradient.tolist() == z_tiny_grad.tolist()
        assert r_gradient.tolist() == r_tiny_grad.tolist()
        assert v_gradient.tolist() == v_tiny_grad.tolist()
        assert w_gradient.tolist() == w_tiny_grad.tolist()
        # print(f"mine:\n", x_gradient.tolist(), f"\ntiny:\n", x_tiny_grad.tolist())
        assert x_gradient.tolist() == x_tiny_grad.tolist()
        assert y_gradient.tolist() == y_tiny_grad.tolist()

    # def test_compare_tinygrad_on_float32(self):
    #     x_val = np.random.random_sample(size=(3, 3)).tolist()
    #     y_val = np.random.random_sample(size=(3, 3)).tolist()
    #
    #     x = Tensor(x_val)
    #     y = Tensor(y_val)
    #     w = x * y
    #     v = w + y
    #     z = v.sum()
    #     z.backward()
    #     z_gradient = z.gradient.ndata
    #     v_gradient = v.gradient.ndata
    #     w_gradient = w.gradient.ndata
    #     x_gradient = x.gradient.ndata
    #     y_gradient = y.gradient.ndata
    #
    #     x_tiny = Tiny_Tensor(x_val, requires_grad=True)
    #     y_tiny = Tiny_Tensor(y_val, requires_grad=True)
    #     w_tiny = x_tiny * y_tiny
    #     v_tiny = w_tiny + y_tiny
    #     z_tiny = v_tiny.sum()
    #     z_tiny.backward()
    #     z_tiny_grad = z_tiny.grad.numpy()
    #     v_tiny_grad = v_tiny.grad.numpy()
    #     w_tiny_grad = w_tiny.grad.numpy()
    #     x_tiny_grad = x_tiny.grad.numpy()
    #     y_tiny_grad = y_tiny.grad.numpy()
    #
    #     print(f"\n{z.ndata.dtype}")
    #     print(z_tiny.numpy().dtype)
    #     print(z.ndata)
    #     print(z_tiny.numpy())
    #     print(f"\n{x_gradient.dtype}")
    #     print(f"{x_tiny_grad.dtype}")
    #     print(f"{x_gradient}")
    #     print(f"{x_tiny_grad}")
    #     assert z.ndata == z_tiny.numpy()
    #     assert x_gradient.dtype == x_tiny_grad.dtype
    #     assert x_gradient.tolist() == x_tiny_grad.tolist()
    #     assert y_gradient.tolist() == y_tiny_grad.tolist()
    #     assert w_gradient.tolist() == w_tiny_grad.tolist()
    #     assert v_gradient.tolist() == v_tiny_grad.tolist()
    #     assert z_gradient.tolist() == z_tiny_grad.tolist()

if __name__ == "__main__":
    unittest.main()
