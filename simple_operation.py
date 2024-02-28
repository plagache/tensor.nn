import numpy as np

from fusion import Tensor

# m = 4096
m = 3

matrice_1 = Tensor(np.random.rand(m, m).astype(np.float32))
matrice_2 = Tensor(np.random.rand(m, m).astype(np.float32))

matrice_3 = matrice_1 + matrice_2
matrice_4 = matrice_3 + matrice_2


print(matrice_1)
print(matrice_2)
print(matrice_3)
print(matrice_4)

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
