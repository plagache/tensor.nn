import unittest
import numpy as np
from fusion import Tensor
from tinygrad.tensor import Tensor as Tiny_Tensor


def test_compare_tinygrad(self, x_np, y_np):
    x = Tensor(x_np)
    y = Tensor(y_np)

    mul = x * y
    add = mul + x
    relu = add.relu()
    sum = relu.sum()
    sum.backward()

    sum_gradient = sum.gradient.ndata
    relu_gradient = relu.gradient.ndata
    add_gradient = add.gradient.ndata
    mul_gradient = mul.gradient.ndata
    x_gradient = x.gradient.ndata
    y_gradient = y.gradient.ndata

    tiny_x = Tiny_Tensor(x_np, requires_grad=True)
    tiny_y = Tiny_Tensor(y_np, requires_grad=True)

    tiny_mul = tiny_x * tiny_y
    tiny_add = tiny_mul + tiny_x
    tiny_relu = tiny_add.relu()
    tiny_sum = tiny_relu.sum()
    tiny_sum.backward()

    tiny_sum_gradient = tiny_sum.grad.numpy()
    tiny_relu_gradient = tiny_relu.grad.numpy()
    tiny_add_gradient = tiny_add.grad.numpy()
    tiny_mul_gradient = tiny_mul.grad.numpy()
    tiny_x_gradient = tiny_x.grad.numpy()
    tiny_y_gradient = tiny_y.grad.numpy()

    # print(np.info(sum.ndata))
    # print(np.info(tiny_sum.numpy()))
    # print(sum.ndata, tiny_sum.numpy())
    # self.assertAlmostEqual(sum.ndata, tiny_sum.numpy(), precision)
    # print(relu.ndata, tiny_relu.numpy())
    # print("precision:", precision)
    # print("decimal:", decimal)
    # np.testing.assert_allclose(sum.ndata, tiny_sum.numpy(), rtol=decimal, atol=decimal)
    np.testing.assert_allclose(sum.ndata, tiny_sum.numpy())
    np.testing.assert_allclose(relu.ndata, tiny_relu.numpy())
    np.testing.assert_allclose(add.ndata, tiny_add.numpy())
    np.testing.assert_allclose(mul.ndata, tiny_mul.numpy())

    # print(f"x_gradient:{x_gradient}\ntiny_x_gradient:{tiny_x_gradient}\n")
    # print(f"\nsum_gradient:\n{sum_gradient}\n----\n{tiny_sum_gradient}")
    # print(f"\nrelu_gradient:\n{relu_gradient}\n----\n{tiny_relu_gradient}")
    # print(f"\nadd_gradient:\n{add_gradient}\n----\n{tiny_add_gradient}")
    # print(f"\nmul_gradient:\n{mul_gradient}\n----\n{tiny_mul_gradient}")
    # print(f"\nx_gradient:\n{x_gradient}\n----\n{tiny_x_gradient}")
    # print(f"\ny_gradient:\n{y_gradient}\n----\n{tiny_y_gradient}")
    np.testing.assert_allclose(sum_gradient, tiny_sum_gradient)
    np.testing.assert_allclose(relu_gradient, tiny_relu_gradient)
    np.testing.assert_allclose(add_gradient, tiny_add_gradient)
    np.testing.assert_allclose(mul_gradient, tiny_mul_gradient)
    np.testing.assert_allclose(x_gradient, tiny_x_gradient)
    np.testing.assert_allclose(y_gradient, tiny_y_gradient)


def test_logarithm(self, x_np, y_np):
    x = Tensor(x_np)
    y = Tensor(y_np)

    tiny_x = Tiny_Tensor(x_np, requires_grad=True)
    tiny_y = Tiny_Tensor(y_np, requires_grad=True)

    logarithm = x.log()
    logarithm.sum().backward()
    x_grad = x.gradient.ndata

    tiny_logarithm = tiny_x.log()
    tiny_logarithm.sum().backward()
    tiny_x_grad = tiny_x.grad.numpy()

    np.testing.assert_allclose(logarithm.ndata, tiny_logarithm.numpy())
    np.testing.assert_allclose(x_grad, tiny_x_grad)


def test_exponantial(self, x_np, y_np):
    x = Tensor(x_np)
    y = Tensor(y_np)

    tiny_x = Tiny_Tensor(x_np, requires_grad=True)
    tiny_y = Tiny_Tensor(y_np, requires_grad=True)

    exponantial = x.exp()
    exponantial.sum().backward()
    x_grad = x.gradient.ndata

    tiny_exponantial = tiny_x.exp()
    tiny_exponantial.sum().backward()
    tiny_x_grad = tiny_x.grad.numpy()

    np.testing.assert_allclose(exponantial.ndata, tiny_exponantial.numpy())
    np.testing.assert_allclose(x_grad, tiny_x_grad)


def test_sigmoid(self, x_np, y_np):
    x = Tensor(x_np)
    y = Tensor(y_np)

    tiny_x = Tiny_Tensor(x_np, requires_grad=True)
    tiny_y = Tiny_Tensor(y_np, requires_grad=True)

    sigmoid = x.sigmoid()
    sigmoid.sum().backward()
    x_grad = x.gradient.ndata
    # print(f"---\nsigmoid = {sigmoid.numpy()}\ndtype(sigmoid)) = {(sigmoid.numpy().dtype)}")
    # print(f"---\nx_grad = {x_grad}\ndtype(x_grad)) = {x_grad.dtype}")

    tiny_sigmoid = tiny_x.sigmoid()
    tiny_sigmoid.sum().backward()
    tiny_x_grad = tiny_x.grad.numpy()
    # print(f"---\ntiny_sigmoid = {tiny_sigmoid.numpy()}\ndtype(tiny_sigmoid) = {(tiny_sigmoid.numpy().dtype)}")
    # print(f"---\ntiny_x_grad = {tiny_x_grad}\ndtype(tiny_x_grad)) = {tiny_x_grad.dtype}")

    np.testing.assert_allclose(sigmoid.ndata, tiny_sigmoid.numpy())
    np.testing.assert_allclose(x_grad, tiny_x_grad)


class test_gradient(unittest.TestCase):
    def test_multiple_type(self):
        x_np = np.random.randint(-9, 9, size=(3, 3))
        y_np = np.random.randint(-9, 9, size=(3, 3))
        # print(x_np)
        # print(y_np)
        # print(np.info(x_np))
        # print(np.info(y_np))
        test_compare_tinygrad(self, x_np, y_np)

        x_np = np.random.uniform(-9, 9, size=(3, 3))
        y_np = np.random.uniform(-9, 9, size=(3, 3))
        test_compare_tinygrad(self, x_np, y_np)

        x_np = np.random.uniform(0, 9, size=(3, 3))
        y_np = np.random.uniform(0, 9, size=(3, 3))
        test_logarithm(self, x_np, y_np)

        x_np = np.random.uniform(-89, 89, size=(3, 3))
        y_np = np.random.uniform(-89, 89, size=(3, 3))
        test_exponantial(self, x_np, y_np)

        x_np = np.random.uniform(-9, 9, size=(3, 3))
        y_np = np.random.uniform(-9, 9, size=(3, 3))
        test_sigmoid(self, x_np, y_np)

        # x_np = np.random.uniform(-9, 9, size=(3, 3))
        # y_np = np.random.uniform(-9, 9, size=(3, 3))
        # test_pow(self, x_np, y_np)


if __name__ == "__main__":
    unittest.main()
