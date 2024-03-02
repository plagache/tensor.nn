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

Data can be of multiples types, and we can do Union of types
in function of the types we can create different Tensor and determine what to apply

Lazy Imperative Programming :
We want to evaluate/compute the gradient only 1 time and all at once
No copy
backpropagation is described as function
for a given shapes/parents/operations

Lazyness :
    - remove copy in profit of memoryview()
    - Tensor should only stored what is nescessary to accomplish the backpropagation
    - we can still display data with numpy function
    - types allow to create the Tensor but
    - shape decide what is the scale of the backward propagation
    - created = Parents
    - creates = children
    - Operations = What type of derivatives should we apply between children and parents
