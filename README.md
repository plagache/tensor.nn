# Fusion
Fusion of different machine learning tools


## Objectives
Create a machine learning framework based on the [micrograd](https://github.com/karpathy/micrograd) of [Andrej Karpathy](https://karpathy.ai/) with the addition of a tensor class.
A Tensor being a multidimensional array, in our case based on numpy ndarray.

- Build small machine learning framework to complete the MLP project of 42
    - [x] dot
    - [ ] softmax
    - [ ] log
        - [ ] how to handle log(0)
- Build and train a small neural network to detect handwritten digits
    - [x] Relu
    - [ ] div
    - [ ] exp
    - [ ] pow
        - how do we not compute gradient for parent that is the power ?

## To-do

I should try to reduce code to a minimum of features,
if you want hardware optimization and lazyness you should use [tinygrad](https://github.com/tinygrad/tinygrad)

- [ ] Sampler class, provide __iter__() methode, return length of batches

- [ ] Tensor class / Automatic differentiation
    - [x] Simple derivatives with context (what created the next value / parent / operation)
    - [x] Backpropagation with topological sort based on Chain of rules
- [ ] Optimizer
    - [ ] GD
    - [ ] SGD
    - [ ] ADAM
- [ ] graphviz of the network / topological sort / important for comprehension of architecture
- [ ] Math
    - [ ] Advanced math function with scikit

- [ ] Module DATA VIZ
    - [ ] Statistics / describe of data
    - [ ] Plot
        - [ ] create a small website with dash or bokeh to plot things
- [ ] Test
    - [ ] what happen if we do operations with another instance than Tensor ? >> we should create a new instance
    - [ ] machine learning operations / mean / logsoftmax / etc
    - [ ] Ensure Type matching (dtype)
    - [ ] different type


## Details

the base for this project to work are 2 class:
- Function:
    - subclass(Function): specify forward and backward for a given operation
    - \*tensors parents: needed for topological sort
    - apply:
        - create & assign \_context
        - return new Tensor based on forward
- Tensor :
    - data : type[numpy array]
    - gradient : type[Tensor]
    - \_context : type[Function]/ops, \*parents tensors

we have made an example in [Functionnement](examples/simple_function.py)

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
