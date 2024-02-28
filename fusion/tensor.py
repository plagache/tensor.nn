from numpy._typing import NDArray


# class Context:
#     def __init__(self, current, parents=(), operation=''):
#         self.current = current
#         self.parents = parents
#         self.operation = operation

# will start with add and multiply
class Tensor:
    def __init__(self, ndarray: NDArray, _children=(), _operation=None):
        self.ndarray = ndarray
        self._previous = set(_children)
        # how can i zero_grad ?
        # i need to match shape for gradient i think
        self.gradient = 0
        self._backward = lambda: None
        self._operation = _operation
        self._shape = self.ndarray.shape

    def __add__(self, other):
        # here the forward, how do we calculate the forward
        output = Tensor(self.ndarray + other.ndarray, (self, other), '+')
        def _backward():
            self.gradient += output.gradient
            other.grad += output.gradient
        return output

    # define topological sort and then apply the _backward for each element in reversed
    # def backward(self):

    def backward(self):

        # The first gradient is always 1.
        self.gradient = 1

        graph_sorted = []
        node_visited = set()
        def topological_sort(node):
            if node not in node_visited:
                node_visited.add(node)
                for parent in node._previous:
                    topological_sort(parent)
                graph_sorted.append(node)

        topological_sort(self)
        # print(*graph_sorted, sep = "\n")

        for node in reversed(graph_sorted):
            # print(node, "\n")
            node._backward()

    def __repr__(self) -> str:
        return f'[Tensor(shape={self._shape}, operation={self._operation}]'
