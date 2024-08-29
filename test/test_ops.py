import unittest
import numpy as np
from fusion import Tensor
from tinygrad.tensor import Tensor as Tiny_Tensor


def test_numpy_type(self):
    list_string = ["one", "two", "three"]
    with self.assertRaises(TypeError) as context:
        test_tensor = Tensor(list_string)
    self.assertIn("Invalid data type", str(context.exception))

    multiple_type = [1, "two", 3.0]
    with self.assertRaises(TypeError) as context:
        test_tensor = Tensor(multiple_type)
    self.assertIn("Invalid data type", str(context.exception))


def test_compare_tinygrad(self, x_np, y_np):
    x = Tensor(x_np)
    y = Tensor(y_np)

    mul = x * y
    add = mul + x
    relu = add.relu()
    sum = relu.sum()
    sum.backward()

    sum_gradient = sum.gradient.data
    relu_gradient = relu.gradient.data
    add_gradient = add.gradient.data
    mul_gradient = mul.gradient.data
    x_gradient = x.gradient.data
    y_gradient = y.gradient.data

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

    np.testing.assert_allclose(sum.data, tiny_sum.numpy())
    np.testing.assert_allclose(relu.data, tiny_relu.numpy())
    np.testing.assert_allclose(add.data, tiny_add.numpy())
    np.testing.assert_allclose(mul.data, tiny_mul.numpy())

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
    x_grad = x.gradient.data

    tiny_logarithm = tiny_x.log()
    tiny_logarithm.sum().backward()
    tiny_x_grad = tiny_x.grad.numpy()

    np.testing.assert_allclose(logarithm.data, tiny_logarithm.numpy())
    np.testing.assert_allclose(x_grad, tiny_x_grad)


def test_exponantial(self, x_np, y_np):
    x = Tensor(x_np)
    y = Tensor(y_np)

    tiny_x = Tiny_Tensor(x_np, requires_grad=True)
    tiny_y = Tiny_Tensor(y_np, requires_grad=True)

    exponantial = x.exp()
    exponantial.sum().backward()
    x_grad = x.gradient.data

    tiny_exponantial = tiny_x.exp()
    tiny_exponantial.sum().backward()
    tiny_x_grad = tiny_x.grad.numpy()

    np.testing.assert_allclose(exponantial.data, tiny_exponantial.numpy())
    np.testing.assert_allclose(x_grad, tiny_x_grad)


def test_sigmoid(self, x_np, y_np):
    x = Tensor(x_np)
    y = Tensor(y_np)

    tiny_x = Tiny_Tensor(x_np, requires_grad=True)
    tiny_y = Tiny_Tensor(y_np, requires_grad=True)

    sigmoid = x.sigmoid()
    sigmoid.sum().backward()
    x_grad = x.gradient.data
    # print(f"---\nsigmoid = {sigmoid.numpy()}\ndtype(sigmoid)) = {(sigmoid.numpy().dtype)}")
    # print(f"---\nx_grad = {x_grad}\ndtype(x_grad)) = {x_grad.dtype}")

    tiny_sigmoid = tiny_x.sigmoid()
    tiny_sigmoid.sum().backward()
    tiny_x_grad = tiny_x.grad.numpy()
    # print(f"---\ntiny_sigmoid = {tiny_sigmoid.numpy()}\ndtype(tiny_sigmoid) = {(tiny_sigmoid.numpy().dtype)}")
    # print(f"---\ntiny_x_grad = {tiny_x_grad}\ndtype(tiny_x_grad)) = {tiny_x_grad.dtype}")

    np.testing.assert_allclose(sigmoid.data, tiny_sigmoid.numpy())
    np.testing.assert_allclose(x_grad, tiny_x_grad)


def test_pow(self, x_np, y_np):
    x = Tensor(x_np)
    y = Tensor(y_np)

    tiny_x = Tiny_Tensor(x_np, requires_grad=True)
    tiny_y = Tiny_Tensor(y_np, requires_grad=True)

    power = x ** Tensor(2)
    # power = x.pow(2)
    power.sum().backward()
    x_grad = x.gradient.data
    print(f"---\npower = {power.numpy()}\ndtype(power)) = {(power.numpy().dtype)}")
    # print(f"---\nx_grad = {x_grad}\ndtype(x_grad)) = {x_grad.dtype}")

    tiny_power = tiny_x**2
    tiny_power.sum().backward()
    tiny_x_grad = tiny_x.grad.numpy()
    print(f"---\ntiny_power = {tiny_power.numpy()}\ndtype(tiny_power) = {(tiny_power.numpy().dtype)}")
    # print(f"---\ntiny_x_grad = {tiny_x_grad}\ndtype(tiny_x_grad)) = {tiny_x_grad.dtype}")

    np.testing.assert_allclose(power.data, tiny_power.numpy())
    np.testing.assert_allclose(x_grad, tiny_x_grad)


class test_gradient(unittest.TestCase):
    def test_multiple_type(self):
        test_numpy_type(self)

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
