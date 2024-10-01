import pytest
from firedrake import *
import torch
from neural_pde.intergrid import torch_interpolation_tensor
from firedrake.ml.pytorch import to_torch

from neural_pde.neural_solver import NeuralSolver

from neural_pde.spherical_patch_covering import SphericalPatchCovering
from neural_pde.patch_encoder import PatchEncoder
from neural_pde.patch_decoder import PatchDecoder
from neural_pde.data_generator import AdvectionDataset
from neural_pde.intergrid import Encoder, Decoder

spherical_patch_covering = SphericalPatchCovering(0, 2)

# number of dynamic fields: scalar tracer
n_dynamic = 1
n_ancillary = 3
latent_dynamic_dim = 7  # picked to hopefully capture the behaviour wanted
latent_ancillary_dim = 3  # also picked to hopefully resolve the behaviour
n_output = 1

num_ref1 = 2
num_ref2 = 3
mesh1 = UnitIcosahedralSphereMesh(num_ref1)  # create the mesh
mesh2 = UnitIcosahedralSphereMesh(num_ref2)
V1 = FunctionSpace(mesh1, "CG", 1)  # define the function space
V2 = FunctionSpace(mesh2, "CG", 1)  # define the function space

### Neural Networks ###
dynamic_encoder_model = torch.nn.Sequential(
    torch.nn.Flatten(start_dim=-2, end_dim=-1),
    torch.nn.Linear(
        in_features=(n_dynamic + n_ancillary)
        * spherical_patch_covering.patch_size,  # size of each input sample
        out_features=latent_dynamic_dim,  # size of each output sample
    ),
).double()  # double means to cast to double precision (float128)

# ancillary encoder model: map ancillary fields to ancillary space
ancillary_encoder_model = torch.nn.Sequential(
    torch.nn.Flatten(start_dim=-2, end_dim=-1),
    torch.nn.Linear(
        in_features=n_ancillary * spherical_patch_covering.patch_size,
        out_features=latent_ancillary_dim,
    ),
).double()

decoder_model = torch.nn.Sequential(
    torch.nn.Linear(
        in_features=latent_dynamic_dim + latent_ancillary_dim,
        out_features=n_output * spherical_patch_covering.patch_size,
    ),
    torch.nn.Unflatten(
        dim=-1, unflattened_size=(n_output, spherical_patch_covering.patch_size)
    ),
).double()

interaction_model = torch.nn.Sequential(
    torch.nn.Flatten(start_dim=-2, end_dim=-1),
    torch.nn.Linear(
        in_features=4
        * (
            latent_dynamic_dim + latent_ancillary_dim
        ),  # do we use a linear model here?? Or do we need a nonlinear part
        out_features=latent_dynamic_dim,
    ),
).double()


def test_trivial2():
    """This tests whether the projection is correct

    PatchEncoder is our projection P. It projects onto the latent space.
    If the PatchDecoder is the Transpose of P, then we must have

    y^T(xP) = (xP)^T(xP) = P^Tx^TY

    and this is what we check."""

    P = torch_interpolation_tensor(fs_from=V1, fs_to=V2, transpose=True)
    X = torch.randn(4, V1.dim()).double()
    Y = torch.matmul(X, P)

    PTXT = torch.matmul(torch.transpose(P, 0, 1), torch.transpose(X, 0, 1))
    PXTY = torch.matmul(PTXT, Y)
    PX2 = torch.matmul(torch.transpose(Y, 0, 1), Y)

    assert torch.allclose(PXTY, PX2)


def test_trivial3():

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

    train_example = AdvectionDataset(V1, 1, 1, 4).__getitem__(0)

    X_old, _ = train_example

    # X_old = torch.randn(4, V1.dim()).double()
    Y = model(X_old)

    X = X_old[0, :]
    Y = Y[0, :]

    XtY = torch.dot(X, Y)
    XtY = XtY.detach().numpy()
    print(XtY)
    # Ax_L2 = torch.linalg.norm(Ax)**2
    Ax = encoder(X_old)
    Ax = Ax[0, :]
    Ax_L2 = torch.dot(Ax, Ax)
    # Ax_L2 = torch.linalg.norm(Ax)**2
    Ax_L2 = Ax_L2.detach().numpy()

    print(Ax_L2**2)
    return np.isclose(XtY, Ax_L2**2)


test_trivial3()


def test_trivial4():

    model = torch.nn.Sequential(
        PatchEncoder(
            V1,
            spherical_patch_covering,
            dynamic_encoder_model,
            ancillary_encoder_model,
            n_dynamic,
        ),
        NeuralSolver(spherical_patch_covering, interaction_model, nsteps=1, stepsize=1),
        PatchDecoder(V1, spherical_patch_covering, decoder_model),
    )

    encoder_model = PatchEncoder(
        V1,
        spherical_patch_covering,
        dynamic_encoder_model,
        ancillary_encoder_model,
        n_dynamic,
    )

    train_example = AdvectionDataset(V1, 1, 1, 4).__getitem__(0)

    X_old, _ = train_example

    # X_old = torch.randn(4, V1.dim()).double()
    Y = model(X_old)

    X = X_old[0, :]
    Y = Y[0, :]

    XtY = torch.dot(X, Y)
    XtY = XtY.detach().numpy()
    print(XtY)
    # Ax_L2 = torch.linalg.norm(Ax)**2
    Ax = encoder_model(X_old)
    Ax = Ax[0, :]
    Ax_L2 = torch.dot(Ax, Ax)
    # Ax_L2 = torch.linalg.norm(Ax)**2
    Ax_L2 = Ax_L2.detach().numpy()

    print(Ax_L2**2)
    return np.isclose(XtY, Ax_L2**2)
