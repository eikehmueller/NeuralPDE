from firedrake import *
import torch

from neural_pde.patch_encoder import PatchEncoder
from neural_pde.patch_decoder import PatchDecoder
from neural_pde.neural_solver import NeuralSolver
from neural_pde.spherical_patch_covering import SphericalPatchCovering

__all__ = ["build_model"]


def build_model(
    n_ref,
    n_func_in_dynamic,
    n_func_in_ancillary,
    n_func_target,
    architecture,
):
    """
    Construct encoder - processor - decoder network

    :arg n_ref: number of refinement steps of icosahedral mesh
    :arg n_func_in_dynamic: number of dynamic input functions
    :arg n_func_in_ancillary: number of ancillary input functions
    :arg n_func_in_target: number of output functions
    :arg architecture: dictionary that describes network architecture
    """
    # construct spherical patch covering
    spherical_patch_covering = SphericalPatchCovering(
        architecture["dual_ref"], architecture["n_radial"]
    )
    print(
        f"  points per patch                = {spherical_patch_covering.patch_size}",
    )
    print(
        f"  number of patches               = {spherical_patch_covering.n_patches}",
    )
    print(
        f"  number of points in all patches = {spherical_patch_covering.n_points}",
    )
    mesh = UnitIcosahedralSphereMesh(n_ref)  # create the mesh
    V = FunctionSpace(mesh, "CG", 1)  # define the function space

    # encoder models
    # dynamic encoder model: map all fields to the latent space
    # input:  (n_dynamic+n_ancillary, patch_size)
    # output: (latent_dynamic_dim)
    dynamic_encoder_model = torch.nn.Sequential(
        torch.nn.Flatten(start_dim=-2, end_dim=-1),
        torch.nn.Linear(
            in_features=(n_func_in_dynamic + n_func_in_ancillary)
            * spherical_patch_covering.patch_size,  # size of each input sample
            out_features=16,
        ),
        torch.nn.Softplus(),
        torch.nn.Linear(in_features=16, out_features=16),
        torch.nn.Softplus(),
        torch.nn.Linear(
            in_features=16,
            out_features=architecture["latent_dynamic_dim"],
        ),
    )

    # ancillary encoder model: map ancillary fields to ancillary space
    # input:  (n_ancillary, patch_size)
    # output: (latent_ancillary_dim)
    ancillary_encoder_model = torch.nn.Sequential(
        torch.nn.Flatten(start_dim=-2, end_dim=-1),
        torch.nn.Linear(
            in_features=n_func_in_ancillary * spherical_patch_covering.patch_size,
            out_features=16,
        ),
        torch.nn.Softplus(),
        torch.nn.Linear(in_features=16, out_features=16),
        torch.nn.Softplus(),
        torch.nn.Linear(
            in_features=16,
            out_features=architecture["latent_ancillary_dim"],
        ),
    )

    # decoder model: map latent variables to variables on patches
    # input:  (d_latent)
    # output: (n_out,patch_size)
    decoder_model = torch.nn.Sequential(
        torch.nn.Linear(
            in_features=architecture["latent_dynamic_dim"]
            + architecture["latent_ancillary_dim"],
            out_features=16,
        ),
        torch.nn.Softplus(),
        torch.nn.Linear(in_features=16, out_features=16),
        torch.nn.Softplus(),
        torch.nn.Linear(
            in_features=16,
            out_features=n_func_target * spherical_patch_covering.patch_size,
        ),
        torch.nn.Unflatten(
            dim=-1,
            unflattened_size=(
                n_func_target,
                spherical_patch_covering.patch_size,
            ),
        ),
    )

    # interaction model: function on latent space
    interaction_model = torch.nn.Sequential(
        torch.nn.Flatten(start_dim=-2, end_dim=-1),
        torch.nn.Linear(
            in_features=4
            * (
                architecture["latent_dynamic_dim"]
                + architecture["latent_ancillary_dim"]
            ),
            out_features=8,
        ),
        torch.nn.Softplus(),
        torch.nn.Linear(
            in_features=8,
            out_features=8,
        ),
        torch.nn.Softplus(),
        torch.nn.Linear(
            in_features=8,
            out_features=architecture["latent_dynamic_dim"],
        ),
    )

    # Full model: encoder + processor + decoder
    model = torch.nn.Sequential(
        PatchEncoder(
            V,
            spherical_patch_covering,
            dynamic_encoder_model,
            ancillary_encoder_model,
            n_func_in_dynamic,
        ),
        NeuralSolver(
            spherical_patch_covering,
            interaction_model,
            nsteps=architecture["nt"],
            stepsize=architecture["dt"],
        ),
        PatchDecoder(V, spherical_patch_covering, decoder_model),
    )
    return model
