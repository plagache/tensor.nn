import numpy as np

# from tinygrad.tensor import Tensor
from fusion import Tensor

# m = 4096
m = 3
t = np.random.rand(m, m).astype(np.float32)

x = Tensor([[1,2,3,4],[5,6,7,8],[9,0,1,2]], need_gradient=False)
y = Tensor([[0,1,0,4],[0,0,1,7],[1,1,0,8]], need_gradient=True)
# x = Tensor([[1,2,3,4],[5,6,7,8],[9,0,1,2]], requires_grad=True)
# y = Tensor([[0,1,0,4],[0,0,1,7],[1,1,0,8]], requires_grad=True)


w = x * y
b = w + y
# print(z)
# z = x.sum()
z = b.sum()
# z = w.sum()
# print(z.shape)
# print(b._context)
# print(z._context)
z.backward()
# print("t.gradient:", t.grad.numpy())
# print("x.gradient:", x.grad.numpy())
# print("y.gradient:", y.grad.numpy())
# print("z.gradient:", z.gradient.ndata)
# print("w.gradient:", w.gradient.ndata)
print("t.gradient:", b.gradient.ndata)
print("x.gradient:", x.gradient.ndata)
print("y.gradient:", y.gradient.ndata)
# print("z.shape:", z.shape)
# print("w.shape:", w.shape)
# print("y.shape:", y.shape)
# print("z.data:", z.ndata)
# print("w.data:", w.ndata)
# print("x.data:", x.ndata)
# print("y.data:", y.ndata)
print("x:", x)
print("y:", y)
