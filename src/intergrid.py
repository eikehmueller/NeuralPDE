"""Intergrid encoder and decoder"""

import torch
from firedrake import *
from firedrake.__future__ import interpolate
from firedrake.ml.pytorch import from_torch, to_torch

__all__ = ["Encoder", "Decoder"]


def torch_interpolation_tensor(fs_from, fs_to, transpose=False):
    """Construct a sparse torch tensor for the interpolation from fs_from to fs_to.

    :arg fs_from: function space to interpolate from
    :arg fs_to: function space to interpolate to
    :arg transpose: transpose matrix?
    """
    u_from = Function(fs_from)
    u_to = Function(fs_to)

    ndof_from = fs_from.dof_count
    ndof_to = fs_to.dof_count

    if transpose:
        mat = np.empty((ndof_from, ndof_to))
    else:
        mat = np.empty((ndof_to, ndof_from))

    for j in range(ndof_from):
        with u_from.dat.vec as v:
            v.set(0)
            v[j] = 1
        u_to = assemble(interpolate(u_from, fs_to))
        with u_to.dat.vec as w:
            if transpose:
                mat[j, :] = w[:]
            else:
                mat[:, j] = w[:]
    a = torch.tensor(mat)
    return a.to_sparse()


class InterpolatorWrapper(torch.autograd.Function):
    """Differentiable PyTorch wrapper around Firedrake interpolation

    Allows interpolation of functions between two Firedrake function
    spaces. Both the forward and adjoint operation are implemented. Since
    interpolation is a linear operation y = A.x, the gradient is propagated
    like grad_in = A^T.grad_out
    """

    @staticmethod
    def interpolate(metadata, x):
        """Interpolate a function given by its dof-vector

        :arg metadata: information on function spaces
        :arg x: dof-vector of function on input function space
        """
        u = from_torch(x, V=metadata["fs_from"])
        w = assemble(action(metadata["interpolator"], u))
        return torch.flatten(to_torch(w))

    @staticmethod
    def adjoint_interpolate(metadata, x):
        """Map a dual vector on the output function space to the input space

        :arg metadata: information on function spaces
        :arg x: dof-vector of dual function on output function space
        """
        w = from_torch(x, V=metadata["fs_to"].dual())
        u = assemble(action(adjoint(metadata["interpolator"]), w))
        return torch.flatten(to_torch(u))

    @staticmethod
    def forward(ctx, metadata, x):
        """Forward map

        Computes y = A.x if x is a dof-vector on the input function space
        or y = A^T.x if x is a dual vector on the output function space

        :arg ctx: context
        :arg metadata: information on function spaces
        :arg x: tensor to operate on
        """
        ctx.metadata.update(metadata)
        if metadata["reverse"]:
            return InterpolatorWrapper.adjoint_interpolate(metadata, x)
        else:
            return InterpolatorWrapper.interpolate(metadata, x)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward map

        :arg ctx: context
        :arg grad_output: gradient with respect out output variables
        """
        if ctx.metadata["reverse"]:
            return None, InterpolatorWrapper.interpolate(ctx.metadata, grad_output)
        else:
            return None, InterpolatorWrapper.adjoint_interpolate(
                ctx.metadata, grad_output
            )


def interpolator_wrapper(fs_from, fs_to, reverse=False):
    """Construct interpolation object from

    Depending on the value of reverse, this either interpolates the
    dof-vectors of functions from fs_from to fs_to or the dof-vectors of
    dual functions from fs_to to fs_from

    :arg fs_from: function space which we want to interpolate from
    :arg fs_to: function space which we want to interpolate to
    """
    interpolator = interpolate(TestFunction(fs_from), fs_to)
    metadata = {
        "fs_from": fs_from,
        "fs_to": fs_to,
        "interpolator": interpolator,
        "reverse": reverse,
    }
    return partial(InterpolatorWrapper.apply, metadata)


class Encoder(torch.nn.Module):
    """Differentiable encoder which interpolates to a different function space

    This maps dof-vectors of functions u from one function space to
    dof-vectors of another function v on another function space in such a way
    that the original function is interpolated to the target space, i.e.
    v = interpolate(u).
    """

    def __init__(self, fs_from, fs_to, assemble=True):
        """Initialise new instance

        :arg fs_from: original function space
        :arg fs_to: target function space that we want to interpolate to
        :arg assemble: assemble sparse torch matrix of interpolation
        """
        super().__init__()
        self.in_features = int(fs_from.dof_count)
        self.out_features = int(fs_to.dof_count)
        self.assemble = assemble
        if self.assemble:
            self.register_buffer(
                "a_sparse", torch_interpolation_tensor(fs_from, fs_to, transpose=True)
            )
        else:
            self._interpolate = interpolator_wrapper(fs_from, fs_to, reverse=False)

    def forward(self, x):
        """Forward pass

        The input will be of shape (batch_size, n_in) and the output of size (batch_size, n_out)
        where n_in and n_out are the dimensions of the function spaces.

        :arg x: input tensor
        """
        if self.assemble:
            return torch.matmul(x, self.a_sparse)
        else:
            return self._forward(x)

    def _forward(self, x):
        """Recursively apply forward pass

        Unbind the input, recursively call the function and stack the outputs.
        If x has dimension 1, apply the interpolation.
        """
        if x.dim() == 1:
            return self._interpolate(x)
        else:
            return torch.stack([self._forward(y) for y in torch.unbind(x)])


class Decoder(torch.nn.Module):
    """Differentiable encoder which implements the adjoint interpolation to a function space

    Maps the dof-vectors of a dual function v* on the target function space
    to the dof-vectors of a dual function u* on the original function space
    in such a way that v = interpolate(u). Hence, this class realises the
    adjoint of the Encoder operation.
    """

    def __init__(self, fs_from, fs_to, assemble=True):
        """Initialise new instance

        :arg fs_from: original function space that we want to interpolate from
        :arg fs_to: target function space to which we want to interpolate
        :arg assemble: assemble sparse torch matrix of interpolation
        """
        super().__init__()
        self.in_features = int(fs_to.dof_count)
        self.out_features = int(fs_from.dof_count)
        self.assemble = assemble
        if self.assemble:
            self.register_buffer(
                "a_sparse", torch_interpolation_tensor(fs_from, fs_to, transpose=False)
            )
        else:
            self._adjoint_interpolate = interpolator_wrapper(
                fs_from, fs_to, reverse=True
            )

    def forward(self, x):
        """Forward pass.

        The input will be of shape (batch_size, n_in) and the output of size (batch_size, n_out)
        where n_in and n_out are the dimensions of the function spaces.

        :arg x: input tensor
        """
        if self.assemble:
            return torch.matmul(x, self.a_sparse)
        else:
            return self._forward(x)

    def _forward(self, x):
        """Recursively apply forward pass

        Unbind the input, recursively call the function and stack the outputs.
        If x has dimension 1, apply the revserve interpolation.
        """
        if x.dim() == 1:
            return self._adjoint_interpolate(x)
        else:
            return torch.stack([self._forward(y) for y in torch.unbind(x)])
