# TODO

Reduce code to a minimum.
I found the solution of [tinygrad](https://github.com/tinygrad/tinygrad) so elegant.

## Autograd

- [ ] Delete and Recreate the tinygrad solution in [examples/tensor.py](examples/tensor.py)

- [ ] Tensor class / Automatic differentiation
    - [x] Simple derivatives with context (what created the next value / parent / operation)
    - [x] Backpropagation with topological sort based on Chain of rules
    - [ ] require_gradient parameters

## Optimizer

- [ ] GD
- [ ] SGD
- [ ] ADAM

## Graph / Data viz

- [ ] graphviz of the network / topological sort / important for comprehension of architecture

- [ ] Module DATA VIZ
    - [ ] Statistics / describe of data
    - [ ] Plot
        - [ ] create a small website with [streamlit](https://github.com/streamlit/streamlit?tab=readme-ov-file)

## Others

- [ ] Math
    - [ ] Advanced math function with scikit

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
