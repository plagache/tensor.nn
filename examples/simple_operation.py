import numpy as np

# from tinygrad.tensor import Tensor
from fusion import Tensor

if __name__ == "__main__":
    # m = 4096
    m = 3
    # t = np.random.rand(m, m).astype(np.float32)
    t = np.random.rand(4, 2).astype(np.float32)

    x = Tensor([[1,2,3,4],[5,6,7,8],[9,0,1,2]], requires_gradient=False)
    y = Tensor([[0,1,0,4],[0,0,1,7],[1,1,0,8]], requires_gradient=True)
    # y = Tensor([0,1,0,4], need_gradient=True)
    # x = Tensor([[1,2,3,4],[5,6,7,8],[9,0,1,2]], requires_grad=True)
    # y = Tensor([[0,1,0,4],[0,0,1,7],[1,1,0,8]], requires_grad=True)


    div = x / y
    div.backward()
    print(div)
    # t = w + y
    # d = t.dot(y.transpose())
    t = Tensor(t)
    # t = t.transpose()
    # w = x * t
    # y = y.transpose()
    # d = y.dot(t)
    d = t.T.dot(y.T)
    r = d.relu()
    z = r.sum()
    print(z)
    # print(z.shape)
    # print(b._context)
    # print(z._context)
    z.backward()
    print(z)
    # print("r.gradient:", r.grad.numpy())
    # print("d.gradient:", d.grad.numpy())
    # print("t.gradient:", t.grad.numpy())
    # print("x.gradient:", x.grad.numpy())
    # print("y.gradient:", y.grad.numpy())
    print("z.gradient:", z.gradient.data)
    print("r.gradient:", r.gradient.data)
    print("d.gradient:", d.gradient.data)
    print("t.gradient:", t.gradient.data)
    print("t.data:", t.data)
    print("y.gradient:", y.gradient.data)
    print("y.data:", y.data)
    # print("w.gradient:", w.gradient.data)
    # print("x.gradient:", x.gradient.data)
    # print("z.shape:", z.shape)
    # print("w.shape:", w.shape)
    # print("y.shape:", y.shape)
    # print("z.data:", z.data)
    # print("w.data:", w.data)
    # print("x.data:", x.data)
    # print("y.data:", y.data)
    print("d:", d)
    print("x:", x)
    print("y:", y)


    s = 0.0004
