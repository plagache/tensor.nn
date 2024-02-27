import numpy as np

from fusion.tensor import Tensor

# m = 4096
m = 256

matrice_1 = Tensor(np.random.rand(m, m).astype(np.float32))
matrice_2 = Tensor(np.random.rand(m, m).astype(np.float32))

matrice_3 = matrice_1 + matrice_2
matrice_4 = matrice_3 + matrice_2


# print(repr(matrice_1))
# print(repr(matrice_2))
print(repr(matrice_3))
print(repr(matrice_4))

# p = 1e-3
#
# a = 2.0
# b = -3.0
# c = 10.0
#
# d1 = a * b + c
#
# print("d1 =", d1)
#
# # a += p
# # b += p
# c += p
#
# d2 = a * b + c
#
# print("d2 =", d2)
#
# print("slope =", (d2 - d1)/p)
