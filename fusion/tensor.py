# class Context:
#     def __init__(self, current, parents=(), operation=''):
#         self.current = current
#         self.parents = parents
#         self.operation = operation

# implement require gradient only in the case of training and backprog
# not nescessary with only inference

# will start with add and multiply
# i need to decompose data in the class
# we can moove the data as a memoryview()
# and display the data with numpy
# class should only stored what is nescessary to accomplish the backprog
# the class describe how the backprog should work

# data can be of multiple type / in 

# type will decide what operation we have to do
# shape
# created
#       operation that created it
# create

import numpy as np
from numpy import dtype
from typing import Tuple, Union

default_type=np.float32

class Tensor:
    def __init__(self, data: Union[int, float, list, np.ndarray], _children=(), _operation=None):
        self.need_gradient : bool = True
        # how can i zero_grad ?
        # i need to match shape for gradient i think
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

        # The first gradient is always 1.
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
