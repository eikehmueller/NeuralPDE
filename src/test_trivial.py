import pytest
from firedrake import *
import torch
from intergrid import Encoder, Decoder
from firedrake.ml.pytorch import to_torch

from neural_pde.neural_solver import Katies_NeuralSolver, NeuralSolver

from neural_pde.spherical_patch_covering import SphericalPatchCovering
from neural_pde.patch_encoder import PatchEncoder
from neural_pde.patch_decoder import PatchDecoder

spherical_patch_covering = SphericalPatchCovering(0, 2)

# number of dynamic fields: scalar tracer
n_dynamic = 1
n_ancillary = 3
latent_dynamic_dim = 7 # picked to hopefully capture the behaviour wanted
latent_ancillary_dim = 3 # also picked to hopefully resolve the behaviour
n_output = 1

num_ref = 2
mesh = UnitIcosahedralSphereMesh(num_ref) # create the mesh
V = FunctionSpace(mesh, "CG", 1) # define the function space
h = 1 / (np.sin(2 * np.pi / 5))# from firedrake documentation
u = Function(V)
n_dofs = len(u.dat.data)

def test_trivial():
    interaction_model = torch.nn.Sequential(
        torch.nn.Flatten(start_dim=-2, end_dim=-1),
        torch.nn.Linear(
            in_features=4 * (latent_dynamic_dim + latent_ancillary_dim), # do we use a linear model here?? Or do we need a nonlinear part
            out_features=latent_dynamic_dim,
        ),
    ).double()

    dynamic_encoder_model = torch.nn.Sequential(
        torch.nn.Flatten(start_dim=-2, end_dim=-1),
        torch.nn.Linear(
            in_features=(n_dynamic + n_ancillary) * spherical_patch_covering.patch_size, # size of each input sample
            out_features=latent_dynamic_dim, # size of each output sample
        ),
    ).double() # double means to cast to double precision (float128)

    # ancillary encoder model: map ancillary fields to ancillary space
    ancillary_encoder_model = torch.nn.Sequential(
        torch.nn.Flatten(start_dim=-2, end_dim=-1), # since we have 2 inputs, this is the same as flattening at 0
        # and this will lead to a completely flatarray
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

    model = torch.nn.Sequential(
        PatchEncoder(
            V,
            spherical_patch_covering,
            dynamic_encoder_model,
            ancillary_encoder_model,
            n_dynamic
            ),
        NeuralSolver(spherical_patch_covering, 
                            interaction_model,
                            nsteps=0, 
                            stepsize=0),
        PatchDecoder(V, spherical_patch_covering, decoder_model),
        )

    X = torch.randn(32, latent_dynamic_dim + latent_ancillary_dim, n_dofs).double()
    print(X.shape)

    Y = model(X)
    return
    #assert torch.allclose(X, Y)
test_trivial()