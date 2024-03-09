import numpy as np

from fusion import Tensor

# m = 4096
m = 3
t = np.random.rand(m, m).astype(np.float32)

x = Tensor([[1,1,0,2],[1,0,1,5],[0,1,1,9]])
# y = Tensor([[0,1,0,4],[0,0,1,7],[1,1,0,8]])
y = Tensor([0,1,0,4])

# p = 1e-6
# a = 4
# b = 7
#
# c1 = a * b
# # a += p
# b += p
# c2 = a * b
# print("slope:", (c2 - c1) / p)
#
#
w = x + y
# print(z)
z = w.sum()
# print(z.shape)
# print(b._context)
# print(z._context)
z.backward()
# print("z.gradient", z.gradient)
# print("w.gradient", w.gradient)
# print("z.data:", z.ndata)
# print("w.data:", w.ndata)
# print("x.data:", x.ndata)
# print("y.data:", y.ndata)
# print("x.gradient:", x.gradient)
# print("y.gradient:", y.gradient)
# print("x:", x)
