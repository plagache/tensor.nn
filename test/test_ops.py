import unittest
from fusion import Tensor

class test_gradient(unittest.TestCase):
    def test_specifique_values(self):
        x = Tensor([[1,2,3,4],[5,6,7,8],[9,0,1,2]])
        y = Tensor([[0,1,0,4],[0,0,1,7],[1,1,0,8]])
        w = x * y
        v = w + y
        z = v.sum()
        z.backward()

        z_gradient = z.gradient.ndata
        assert z_gradient == 1
        assert z.shape == ()

        v_gradient = v.gradient.ndata
        assert v_gradient.tolist() == [[1,1,1,1],[1,1,1,1],[1,1,1,1]]
        assert v_gradient.shape == (3, 4)

        w_gradient = w.gradient.ndata
        assert w_gradient.tolist() == [[1,1,1,1],[1,1,1,1],[1,1,1,1]]
        assert w_gradient.shape == (3, 4)

        x_gradient = x.gradient.ndata
        assert x_gradient.tolist() == [[0,1,0,4],[0,0,1,7],[1,1,0,8]]
        assert x_gradient.shape == (3, 4)

        y_gradient = y.gradient.ndata
        assert y_gradient.tolist() == [[2,3,4,5],[6,7,8,9],[10,1,2,3]]
        assert y_gradient.shape == (3, 4)

if __name__ == '__main__':
    unittest.main()
