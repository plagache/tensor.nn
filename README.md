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

the base for this to work is the Function class that create:
- context : parents, type[Function] / ops
we have made an example in [Functionnement](examples/simple_function)

Data can be of multiples types, and we can do Union of types
in function of the types we can create different Tensor and determine what to apply

Lazy Imperative Programming :
We want to evaluate/compute the gradient only 1 time and all at once
No copy
backpropagation is described as function
for a given shapes/parents/operations

Lazyness :
    - remove copy in profit of memoryview()
    - create a kernel that represent all the operations / load for one pass
