# Fusion
Fusion of different machine learning tools


## Objectives
Create a machine learning framework based on the [micrograd](https://github.com/karpathy/micrograd) of [Andrej Karpathy](https://karpathy.ai/) with the addition of a tensor class based on numpy.


## ToDo

- [ ] Tensor class / Automatic differentiation
    - [ ] Simple derivatives with context (what created the next value / parent / operation)
    - [ ] Backpropagation with topological sort based on Chain of rules
    - [ ] lazyness
- [ ] Optimizer
    - [ ] GD
    - [ ] SGD
    - [ ] ADAM
- [ ] Preprocessing
    - [ ] pandas
    - [ ] datasets
    - [ ] hugging face
- [ ] graphviz of the network / topological sort
- [ ] Math
    - [ ] Advenced math function with scikit
- [ ] Statistics / describe of data
- [ ] Plot


## Details

the base for this project to work are 2 class:
- Function:
    - subclass(Function): specify forward and backward for a given ops
    - \*tensors parents: needed for topological sort
    - apply:
        - create & assign \_context
        - return new Tensor based on forward
- Tensor :
    - data
    - gradient
    - \_context : type[Function]/ops, \*parents tensors

we have made an example in [Functionnement](examples/simple_function.py)

Data can be of multiples types, and we can do Union of types

Lazyness :
    - remove copy/tensor creation for each node in profit of memoryview()
    - create a kernel that represent all the operations to compute the needed gradient / load for one pass

### operations

#### machine learning Ops to calculate derivatives

```
Relu, Log, Exp                                 # unary ops
Sum, Max                                       # reduce ops (with axis argument)
Maximum, Add, Sub, Mul, Pow, Div, Equal        # binary ops (no broadcasting, use expand)
Expand, Reshape, Permute, Pad, Shrink, Flip    # movement ops
```


#### low level Ops
these are cpu_features relative to the raw operations of your hardware
