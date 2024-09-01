# Fusion
Fusion of different machine learning tools


## Objectives
Understand machine learning framework and autograd with the [micrograd](https://github.com/karpathy/micrograd) of [Andrej Karpathy](https://karpathy.ai/), and then add the tensor class.
A Tensor containing a multidimensional array, in our case based on numpy ndarray.

- Build small machine learning framework to complete the MLP project of 42
    - [x] dot
    - [ ] softmax
    - [x] sigmoid
    - [x] log
- Build and train a small neural network to detect handwritten digits
    - [x] Relu
    - [x] div
    - [x] exp
    - [ ] pow
        - [ ] * itself ^N times

## To-do

I should try to reduce code to a minimum.
I found the solution of [tinygrad](https://github.com/tinygrad/tinygrad) so elegant.

"what i cannot create, i do not understand" - [Richard Feynman](https://en.wikipedia.org/wiki/Richard_Feynman)

- [ ] Delete and Recreate the tinygrad solution in [examples/tensor.py](examples/tensor.py)

- [ ] Tensor class / Automatic differentiation
    - [x] Simple derivatives with context (what created the next value / parent / operation)
    - [x] Backpropagation with topological sort based on Chain of rules
    - [ ] require_gradient parameters
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

- [ ] Sampler class, provide __iter__() methode, return length of batches

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
    - gradient : type[numpy array]
    - \_context : type[Function]/ops, takes \*parents tensors as arguments
