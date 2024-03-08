import numpy as np

from fusion import Tensor

# m = 4096
m = 3
t = np.random.rand(m, m).astype(np.float32)

x = Tensor([[1,1,0,2],[1,0,1,5],[0,1,1,9]])
y = Tensor([[0,1,0,4],[0,0,1,7],[1,1,0,8]])

b = x + y
# print(z)
z = b.sum()
# print(x.ndata)
# print(y.ndata)
# print(z.ndata)
# print(z.ndata)
# print(z.shape)
# print(b._context)
# print(z._context)
z.backward()
print("z.gradient", z.gradient)
print("b.gradient", b.gradient)
print("x.gradient", x.gradient)
print("y.gradient", y.gradient)
# print(np.array(1))
# print(b.gradient)
# print(x.gradient)
