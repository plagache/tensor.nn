# class Context:
#     def __init__(self, current, parents=(), operation=''):
#         self.current = current
#         self.parents = parents
#         self.operation = operation


from typing import Optional, Tuple, Union

import numpy as np
from numpy import dtype

default_type=np.float32

# it has a Function and a tensor
# Tensor has the data and the shape
# Function take a variables number of Tensor and describe the forward and backward pass
# Forward create context attach to Tensors / parents / ops
# Backward describe how the gradient is populated

class Function:
    def __init__(self, *tensors:Tensor):
        self.parents = tensors

        # Context: internal variables used for autograd graph construction
        # maybe create a new class for this
        # self._parents = set(_children)
        # self._backward = lambda: None
        # self._operation = _operation

    # *args = positional arguments
    # **kwargs = keyword arguments
    def forward(self, *args, **kwargs): raise NotImplementedError("forward not implemented")
    def backward(self, *args, **kwargs): raise NotImplementedError("backward not implemented")

    # @classmethod provides a way to define methods that operate on the class itself rather than instances of the class.
    @classmethod
    def apply(function:Type[Function]):
        # context is on the class level not the instance level
        return

class Tensor:
    def __init__(self, data: Union[int, float, list, np.ndarray], _children=(), _operation=None):
        # True for training / False for inference
        self.need_gradient : bool = True

        # to zero_grad / maybe we don't need / we should just assume zero is always the case
        # we create a copy in shape of the tensor and zeroed all the value
        self.gradient : Optional[Tensor] = None

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

    # def numpy(self) -> np.ndarray: return self.ndata
    # def numpy(self) -> np.ndarray: return self.lazydata.numpy()


    def topological_sort(self):
        def _topological_sort(node, node_visited, graph_sorted):
            if node not in node_visited:
                node_visited.add(node)
                for parent in node._parents:
                    _topological_sort(parent, node_visited, graph_sorted)
                graph_sorted.append(node)
            return graph_sorted
        return _topological_sort(self, set(), [])


    # Create new Tensor at each node with the gradients
    def backward(self):

        # First gradient is always one
        self.gradient = Tensor(1)

        # for each node we want te create a Parents Tensor Gradients
        for node in reversed(self.topological_sort()):
            gradients = node._backward()

    # def sum(self):

    # def __add__(self, other):
 

    def __repr__(self) -> str:
        # assert self.ndata.shape is not None
        # return f'[Tensor(shape={self.ndata.shape}, operation={self._operation}]'
        # we do not store operation on the Tensor: its in Function
        return f'<Tensor(ndata={self.ndata}>'
