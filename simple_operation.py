import numpy as np

from fusion import Tensor

# m = 4096
m = 3
t = np.random.rand(m, m).astype(np.float32)

x = Tensor([[1,1,0],[1,0,1],[0,1,1]])
y = Tensor([[0,1,0],[0,0,1],[1,1,0]])
# y = Tensor([[2.0,0,-2.0]])

matrice_1 = Tensor(t)
matrice_2 = Tensor(t)
matrice_3 = matrice_1 + matrice_2
# matrice_4 = matrice_3 + matrice_2

z = y + x
# print(x.shape)
# z = x.sum()
print(z)
print(z.shape)
z.backward()

print(z)
print(z.dtype)
print(z.shape)
print(z.numpy())
print(x.gradient)
print(y.gradient)
print(z.gradient)
# print(x)
# print(y)
# print(y.dtype)
# print(x.dtype)
# print(y.shape)
# print(x.shape)
# print(x.ndata)
# print(y.ndata)

# print(matrice_1)
# print(matrice_2)
# print(matrice_3)
# print(matrice_4)
# print(matrice_4.gradient)

# from tinygrad import Tensor
#
# x = Tensor.eye(3, requires_grad=True)
# print(x.numpy())
# print(x)
# y = Tensor([[2.0,0,-2.0]], requires_grad=True)
# print(y.numpy())
# # z = (y + x).mean()
# # z = (y * x).sum()
# z = y.matmul(x).sum()
# print(z.numpy())
# z.backward()
# #
# print(x.grad.numpy())  # dz/dx
# print(y.grad.numpy())  # dz/dy
