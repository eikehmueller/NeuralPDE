"""This module tests the trivial cases where nt = 0. The tests here are

1: check that the projection matrix satisfies trivial matrix permutations
2: check that  model = decoder(encoder(.)) maps X to X."""

import sys
sys.path.append("..")
from firedrake import *
import torch
from neural_pde.intergrid import torch_interpolation_tensor
from neural_pde.spherical_patch_covering import SphericalPatchCovering
from neural_pde.patch_encoder import PatchEncoder
from neural_pde.patch_decoder import PatchDecoder

spherical_patch_covering = SphericalPatchCovering(0, 0)

n_dynamic = 1  # number of dynamic fields: scalar tracer
n_ancillary = 0  # number of ancillary fields: x-, y- and z-coordinates
n_output = 1  # number of output fields: scalar tracer

num_ref1 = 1
num_ref2 = 1

mesh1 = UnitIcosahedralSphereMesh(num_ref1)  # create the mesh
mesh2 = UnitIcosahedralSphereMesh(num_ref2)


V1 = FunctionSpace(mesh1, "CG", 1)  # define the function space
V2 = FunctionSpace(mesh2, "CG", 1)  # define the function space


dynamic_encoder_model = torch.nn.Flatten(start_dim=-2, end_dim=-1)
ancillary_encoder_model = torch.nn.Flatten(start_dim=-2, end_dim=-1)
decoder_model = torch.nn.Unflatten(
    dim=-1, unflattened_size=(n_output, spherical_patch_covering.patch_size)
)


def test_trivial_projection():
    """Check that the projection matrix satisfies trivial matrix operations

    PatchEncoder is our projection P.  If the PatchDecoder is the Transpose of P,
    then y^T(xP) = (xP)^T(xP) = P^Tx^TY and this is what is checked."""

    # construct P, X and Y
    batchsize = 4
    P = torch_interpolation_tensor(fs_from=V1, fs_to=V2, transpose=True)
    X = torch.randn(batchsize, V1.dim())
    Y = torch.matmul(X, P)

    # calculate the transpose
    PTXT = torch.matmul(torch.transpose(P, 0, 1), torch.transpose(X, 0, 1))
    PXTY = torch.matmul(PTXT, Y)
    PX2 = torch.matmul(torch.transpose(Y, 0, 1), Y)

    assert torch.allclose(PXTY, PX2)


def test_trivial_encoder():
    """Check that model = decoder(encoder(.)) maps X to X

    For random input X compute Y = model(X) and Ax = encoder(X). Then check that
    the dot-product X^T.Y is identical to the squared l2 norm of Ax, i.e. ||Ax||_2^2 = Ax^T.Ax.
    """

    # define the model
    encoder = PatchEncoder(
        V1,
        spherical_patch_covering,
        dynamic_encoder_model,
        ancillary_encoder_model,
        n_dynamic,
    )
    decoder = PatchDecoder(V1, spherical_patch_covering, decoder_model)

    model = torch.nn.Sequential(
        encoder,
        decoder,
    )

    batchsize = 4
    X = torch.randn(batchsize, n_dynamic + n_ancillary, V1.dim())
    Y = model(X)

    # calculate XtY
    XtY = torch.einsum("bij,bij->bi", X, Y)
    XtY = XtY.detach().numpy()

    # calculate l2 norm of Ax
    Ax = encoder(X)
    Ax_L2 = torch.einsum("bpi,bpi->bi", Ax, Ax)
    Ax_L2 = Ax_L2.detach().numpy()

    assert np.all(np.isclose(XtY, Ax_L2))
