from __future__ import annotations

from typing import Optional, Tuple, Type, Union

import numpy as np
from numpy import dtype

default_type = np.float32

# it has a Function and a tensor
# Tensor has the data and the shape
# Function take a variables number of Tensor and describe the forward and backward pass
# Forward create context attach to Tensors / parents / ops
# Backward describe how the gradient is populated


class Function:
    def __init__(self, *tensors: Tensor):
        self.parents = tensors

    # *args = positional arguments
    # **kwargs = keyword arguments
    def forward(self, *args):
        raise NotImplementedError("forward not implemented")

    def backward(self, *args):
        raise NotImplementedError("backward not implemented")

    # @classmethod provides a way to define methods that operate on the class itself rather than instances of the class.
    # The apply method is used to instantiate and execute the forward pass of the function, returning a tensor representing the result.
    @classmethod
    def apply(function: Type[Function], *x: Tensor):
        context = function(*x)
        output = Tensor(context.forward(*x))
        output._context = context
        return output


class Add(Function):
    def forward(self, x: Tensor, y: Tensor):
        return np.add(x.ndata, y.ndata)

    def backward(self, output: Tensor):
        return output, output


class Sum(Function):
    def forward(self, x: Tensor):
        self.input_shape = x.shape
        return np.sum(x.ndata)

    def backward(self, output: Tensor):
        return np.broadcast_to(output.ndata, self.input_shape)


class Tensor:
    def __init__(self, data: Union[int, float, list, np.ndarray]):
        # True for training / False for inference
        self.need_gradient: bool = True

        # to zero_grad / maybe we don't need / we should just assume zero is always the case
        # we create a copy in shape of the tensor and zeroed all the value
        self.gradient: Optional[Tensor] = None

        # Context: internal variables used for autograd graph construction
        # _ mean private context
        self._context: Optional[Function] = None

        if isinstance(data, (int, float, np.integer, list)):
            self.ndata = np.array(data)
            return

        if isinstance(data, np.ndarray):
            self.ndata = data
            return

    # @property is just a getter() | in our case it gets the shape()
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.ndata.shape

    @property
    def dtype(self) -> dtype:
        return self.ndata.dtype

    # def dtype(self) -> dtype: return np.dtype(self.ndata)

    # def numpy(self) -> np.ndarray: return self.ndata
    # def numpy(self) -> np.ndarray: return self.lazydata.numpy()

    def topological_sort(self):
        def _topological_sort(node, node_visited, graph_sorted):
            if node not in node_visited:
                if getattr(node, "_context", None):
                    node_visited.add(node)
                    for parent in node._context.parents:
                        _topological_sort(parent, node_visited, graph_sorted)
                        # yield / is for lazy, and yield the data
                    graph_sorted.append(node)
            return graph_sorted

        return _topological_sort(self, set(), [])

    # Create new Tensor at each node with the gradients
    def backward(self):
        # First gradient is always one
        self.gradient = Tensor(1)

        # for each node we want te create a Parents Tensor Gradients
        # print("\nTopo sort:", self.topological_sort())
        # print("\nself:", self)
        # print("\nbefore Backward =================")
        # print("\nself.gradient:", self.gradient)
        for node in reversed(self.topological_sort()):
            assert node.gradient is not None
            print("\n=============== New Node =================")
            for parents in node._context.parents:
                # print(f"parents len:{len(node._context.parents)}")
                # print(f"parents: {parents}")
                print(f"node: {node}\ngradient: {node.gradient}\ncontexte: {node._context}\nbackward: {node._context.backward(node.gradient)}\nEnd ==================")
                gradients = node._context.backward(node.gradient)
                # print(f"gradients:{gradients}")
                # gradients = [Tensor(gradients)]
                if len(node._context.parents) == 1:
                    parents.gradient = gradients
                else:
                    for parent, gradient in zip(parents, gradients):
                        assert gradient.shape == parent.shape, f"grad shape must match tensor shape, {gradient.shape!r} != {parent.shape!r}"
                        parent.gradient = gradient
                # we have multiple parents and need to match backward result with shape
            del node._context
        return self

    def add(self, other):
        return Add.apply(self, other)

    def __add__(self, other):
        return self.add(other)

    def sum(self):
        return Sum.apply(self)

    # def __sum__(self):
    #     return self.sum()

    def __repr__(self) -> str:
        # assert self.ndata.shape is not None
        # return f'<Tensor(shape={self.ndata.shape})>'
        # we do not store operation on the Tensor: its in Function
        return f"<Tensor(ndata={self.ndata})>"
