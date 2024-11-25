from typing import Optional
import numpy as np


class Operation:
    def __init__(self, *parents):
        self.parents = parents

    def forward(self, *args):
        raise NotImplementedError()

    def backward(self, *args):
        raise NotImplementedError()

    @classmethod
    def apply(cls, *inputs):
        operation: Operation = cls(*inputs)
        output = Tensor(operation.forward(inputs))
        output._context = operation
        return output



class Add(Operation):
    def forward(self, x, y):
        return np.add(x.data, y.data)

    def backward(self, output):
        x, y = self.parents
        return output.data, output.data

class Tensor:
    def __init__(self, data):
        self.data = data
        self.gradient: Optional[Tensor] = None
        self._context: Optional[Operation] = None

    def __add__(self, other):


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
        print("\nmul result:", mul.data, tiny_mul.numpy())
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
