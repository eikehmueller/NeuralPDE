## Neural PDE

[Firedrake](https://www.firedrakeproject.org/)/[PyTorch](https://pytorch.org/) implementation of a neural surrogate model for time-dependent PDEs on the sphere. The structure to the model is similar to the one in [Ryan Keisler's paper](https://arxiv.org/abs/2202.07575) and [GraphCast](https://www.science.org/doi/epdf/10.1126/science.adi2336). However, the processor solves a time-dependent ODE instead of using message passing on a Graph Neural network. Hence, the model is a realisation of a [Neural ODE](https://arxiv.org/abs/1806.07366).

![Model structure](figures/model_structure.svg)
*Figure 1: model structure*


### Mathematical description
For a mathematical description see [here](Description.ipynb)

### Installation
To install this package run 
```python -m pip install .```
as usual after installing the dependencies (see below).

If you want to edit the code, you might prefer to install in editable mode with
```python -m pip install --editable .```

### Dependencies
#### Firedrake
See [here](https://www.firedrakeproject.org/download.html) for Firedrake installation instructions.
#### PyTorch
This should be automatically installed when running `pip` (see above). However, you will likely first have to set up CUDA etc to be able to run with GPU support.
