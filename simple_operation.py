import numpy as np

from fusion import Tensor

# m = 4096
m = 3
t = np.random.rand(m, m).astype(np.float32)

x = Tensor([[1,1,0],[1,0,1],[0,1,1]])
y = Tensor([[0,1,0],[0,0,1],[1,1,0]])

# z = x + y
# print(z)
z = x.sum()
print(z.shape)
# z.backward()
