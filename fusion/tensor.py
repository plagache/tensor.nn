# class Context:
#     def __init__(self, current, parents=(), operation=''):
#         self.current = current
#         self.parents = parents
#         self.operation = operation


from typing import Tuple, Union

import numpy as np
from numpy import dtype

default_type=np.float32

class Tensor:
    def __init__(self, data: Union[int, float, list, np.ndarray], _children=(), _operation=None):
        # True for training / False for inference
        self.need_gradient : bool = True

        # to zero_grad
        # we create a copy in shape of the tensor and zeroed all the value
        # self.gradient = Tensor(0)
        self.gradient = 0

        # Context: internal variables used for autograd graph construction
        # maybe create a new class for this
        self._parents = set(_children)
        self._backward = lambda: None
        self._operation = _operation


        if isinstance(data, (int, float)):
            self.ndata = np.array(data)
            return

        if isinstance(data, list):
            self.ndata = np.array(data)
            return

        if isinstance(data, np.ndarray):
            self.ndata = data
            return

    # @property is just a getter() | in our case it gets the shape()
    @property
    def shape(self) -> Tuple[int, ...]: return self.ndata.shape

    @property
    def dtype(self) -> dtype: return self.ndata.dtype
    # def dtype(self) -> dtype: return np.dtype(self.ndata)

    def numpy(self) -> np.ndarray: return self.ndata
    # def numpy(self) -> np.ndarray: return self.lazydata.numpy()

    # define topological sort and then apply the _backward for each element in reversed
    # def backward(self):

    def topological_sort(self):
        def _topological_sort(node, node_visited, graph_sorted):
            if node not in node_visited:
                node_visited.add(node)
                for parent in node._parents:
                    _topological_sort(parent, node_visited, graph_sorted)
                graph_sorted.append(node)
            return graph_sorted
        return _topological_sort(self, set(), [])


    def backward(self):

        # The first gradient should always be 1.
        # And a scalar value / Loss calculated : 0.80 something
        # in a tensor and backpropagate to the biggest shape() of inputs X
        self.gradient = 1
        # self.gradient = Tensor(1)

        # self.topological_sort()
        # topological_sort(self)
        # print(*graph_sorted, sep = "\n")

        for node in reversed(self.topological_sort()):
            # print(node, "\n")
            node._backward()
            # self.gradient = node._backward()

    def __add__(self, other):
        output = Tensor(self.ndata + other.ndata, (self, other), '+')
        def _backward():
            self.gradient += output.gradient
            other.gradient += output.gradient
        output._backward = _backward
        return output
        # return self.ndata + other.ndata

    def __repr__(self) -> str:
        # assert self.ndata.shape is not None
        # return f'[Tensor(shape={self.ndata.shape}, operation={self._operation}]'
        return f'<Tensor(ndata={self.ndata}, operation={self._operation}>'
