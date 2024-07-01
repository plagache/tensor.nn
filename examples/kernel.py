from tinygrad.tensor import Tensor
from fusion import Tensor as FTensor

x = [[1,2,3,4],[5,6,7,8],[9,0,1,2]]
y = [[0,1,0,4],[0,0,1,7],[1,1,0,8]]

a = Tensor(x)
b = Tensor(y)
res = a.dot(b.T).numpy()
print(res)

fx = FTensor(x)
fy = FTensor(y)
f_res = fx.dot(fy.T)
print(f_res)
