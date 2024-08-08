from functools import partial
import torch
import numpy as np


class LinearOperator(torch.autograd.Function):
    """Implementation of y=A.x"""

    @staticmethod
    def forward(ctx, metadata, *xP):
        """Forward pass"""
        ctx.metadata.update(metadata)
        return x.detach() @ metadata["A"]

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output.detach() @ ctx.metadata["A"].T


def linear_operator(A):
    metadata = {"A": A}
    return partial(LinearOperator.apply, metadata)


batch_size = 4
n_in = 7
n_out = 2

rng = np.random.default_rng(seed=41517)
A = rng.normal(size=(n_in, n_out))

model = linear_operator(A)
x = torch.tensor(rng.normal(size=(batch_size, n_in)), requires_grad=True)
y = model(x)
z = torch.sum(y)
z.backward()
print(x.grad)
print(np.sum(A, axis=1))
