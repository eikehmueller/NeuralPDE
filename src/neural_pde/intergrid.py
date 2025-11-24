"""Intergrid encoder and decoder. Contains classes that define
differentiable interpolation functions from one function space
to another."""

import torch
from firedrake import *
from firedrake.__future__ import interpolate

__all__ = ["Interpolator", "AdjointInterpolator"]


def torch_interpolation_tensor(fs_from, fs_to, transpose=False, dtype=None):
    """Construct a sparse torch tensor for the interpolation from fs_from to fs_to.

    :arg fs_from: function space to interpolate from
    :arg fs_to: function space to interpolate to
    :arg transpose: transpose matrix?
    :arg dtype: datatype. Use torch default if None
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
    a = torch.tensor(mat, dtype=torch.get_default_dtype() if dtype is None else dtype)
    return a.to_sparse()


class Interpolator(torch.nn.Module):
    """Differentiable encoder which interpolates to a different function space

    This maps dof-vectors of functions u from one function space to
    dof-vectors of another function v on another function space in such a way
    that the original function is interpolated to the target space, i.e.
    v = interpolate(u).
    """

    def __init__(self, fs_from, fs_to, dtype=None):
        """Initialise new instance

        :arg fs_from: original function space
        :arg fs_to: target function space that we want to interpolate to
        :arg dtype: datatype. Use torch default if None
        """
        super().__init__()
        self.in_features = int(fs_from.dof_count)
        self.out_features = int(fs_to.dof_count)
        self.register_buffer(
            "a_sparse",
            torch_interpolation_tensor(
                fs_from,
                fs_to,
                transpose=True,
                dtype=torch.get_default_dtype() if dtype is None else dtype,
            ),
        )

    def forward(self, x):
        """Forward pass

        The input will be of shape (batch_size, n_in) and the output of size (batch_size, n_out)
        where n_in and n_out are the dimensions of the function spaces.

        :arg x: input tensor
        """
        return torch.matmul(x, self.a_sparse)


class AdjointInterpolator(torch.nn.Module):
    """Differentiable encoder which implements the adjoint interpolation to a function space

    Maps the dof-vectors of a dual function v* on the target function space
    to the dof-vectors of a dual function u* on the original function space
    in such a way that v = interpolate(u). Hence, this class realises the
    adjoint of the Encoder operation.
    """

    def __init__(self, fs_from, fs_to, dtype=None):
        """Initialise new instance

        :arg fs_from: original function space that we want to interpolate from
        :arg fs_to: target function space to which we want to interpolate
        :arg dtype: datatype. Use torch default if None
        """
        super().__init__()
        self.in_features = int(fs_to.dof_count)
        self.out_features = int(fs_from.dof_count)
        self.register_buffer(
            "a_sparse",
            torch_interpolation_tensor(
                fs_from,
                fs_to,
                transpose=False,
                dtype=torch.get_default_dtype() if dtype is None else dtype,
            ),
        )

    def forward(self, x):
        """Forward pass.

        The input will be of shape (batch_size, n_in) and the output of size (batch_size, n_out)
        where n_in and n_out are the dimensions of the function spaces.

        :arg x: input tensor
        """

        return torch.matmul(x, self.a_sparse)
