"""Encoder-processor-decoder model"""

from firedrake import *
import torch
import os
import json

from neural_pde.patch_encoder import PatchEncoder
from neural_pde.patch_decoder import PatchDecoder
from neural_pde.neural_solver import NeuralSolver
from neural_pde.spherical_patch_covering import SphericalPatchCovering

__all__ = ["build_model", "load_model"]


def build_model(
    n_ref,
    n_func_in_dynamic,
    n_func_in_ancillary,
    n_func_target,
    architecture,
):
    """
    Construct encoder - processor - decoder model

    :arg n_ref: number of refinement steps of icosahedral mesh
    :arg n_func_in_dynamic: number of dynamic input functions
    :arg n_func_in_ancillary: number of ancillary input functions
    :arg n_func_in_target: number of output functions
    :arg architecture: dictionary that describes network architecture
    """
    model = NeuralPDEModel()
    model.setup(
        n_ref, n_func_in_dynamic, n_func_in_ancillary, n_func_target, architecture
    )
    return model


def load_model(directory):
    """Load model from disk

    :arg directory: directory containing the saved model"""
    model = NeuralPDEModel()
    model.load(directory)
    return model


class NeuralPDEModel(torch.nn.Sequential):
    """Class representing the encoder - processor - decoder network"""

    def __init__(self):
        """Initialise a new instance with empty model"""
        super().__init__()
        self.architecture = None
        self.dimensions = None
        self.initialised = False

    def setup(
        self, n_ref, n_func_in_dynamic, n_func_in_ancillary, n_func_target, architecture
    ):
        """
        Initialise new instance with model

        :arg n_ref: number of refinement steps of icosahedral mesh
        :arg n_func_in_dynamic: number of dynamic input functions
        :arg n_func_in_ancillary: number of ancillary input functions
        :arg n_func_in_target: number of output functions
        :arg architecture: dictionary that describes network architecture
        """
        assert not self.initialised
        self.architecture = architecture
        self.dimensions = dict(
            n_ref=n_ref,
            n_func_in_dynamic=n_func_in_dynamic,
            n_func_in_ancillary=n_func_in_ancillary,
            n_func_target=n_func_target,
        )
        # construct spherical patch covering
        spherical_patch_covering = SphericalPatchCovering(
            architecture["dual_ref"], architecture["n_radial"]
        )
        print(
            f"  points per patch                     = {spherical_patch_covering.patch_size}",
        )
        print(
            f"  number of patches                    = {spherical_patch_covering.n_patches}",
        )
        print(
            f"  number of points in all patches      = {spherical_patch_covering.n_points}",
        )
        mesh = UnitIcosahedralSphereMesh(n_ref)  # create the mesh
        V = FunctionSpace(mesh, "CG", 1)  # define the function space
        print(f"  number of unknowns of function space = {V.dof_count}")

        # encoder models
        # dynamic encoder model: map all fields to the latent space
        # input:  (n_dynamic+n_ancillary, patch_size)
        # output: (latent_dynamic_dim)
        n_hidden = 64
        dynamic_encoder_model = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=-2, end_dim=-1),
            torch.nn.Linear(
                in_features=(n_func_in_dynamic + n_func_in_ancillary)
                * spherical_patch_covering.patch_size,  # size of each input sample
                out_features=n_hidden,
            ),
            torch.nn.Softplus(),
            torch.nn.Linear(in_features=n_hidden, out_features=n_hidden),
            torch.nn.Softplus(),
            torch.nn.Linear(in_features=n_hidden, out_features=n_hidden),
            torch.nn.Softplus(),
            torch.nn.Linear(
                in_features=n_hidden,
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
                out_features=n_hidden,
            ),
            torch.nn.Softplus(),
            torch.nn.Linear(in_features=n_hidden, out_features=n_hidden),
            torch.nn.Softplus(),
            torch.nn.Linear(in_features=n_hidden, out_features=n_hidden),
            torch.nn.Softplus(),
            torch.nn.Linear(
                in_features=n_hidden,
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
                out_features=n_hidden,
            ),
            torch.nn.Softplus(),
            torch.nn.Linear(in_features=n_hidden, out_features=n_hidden),
            torch.nn.Softplus(),
            torch.nn.Linear(in_features=n_hidden, out_features=n_hidden),
            torch.nn.Softplus(),
            torch.nn.Linear(
                in_features=n_hidden,
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
                out_features=16,
            ),
            torch.nn.Softplus(),
            torch.nn.Linear(
                in_features=16,
                out_features=architecture["latent_dynamic_dim"],
            ),
        )

        # Full model: encoder + processor + decoder
        self.add_module(
            "PatchEncoder",
            PatchEncoder(
                V,
                spherical_patch_covering,
                dynamic_encoder_model,
                ancillary_encoder_model,
                n_func_in_dynamic,
            ),
        )
        self.add_module(
            "NeuralSolver",
            NeuralSolver(
                spherical_patch_covering,
                interaction_model,
                nsteps=architecture["nt"],
                stepsize=architecture["dt"],
            ),
        )
        self.add_module(
            "PatchDecoder",
            PatchDecoder(V, spherical_patch_covering, decoder_model),
        )
        self.initialised = True

    def save(self, directory):
        """Save model to disk

        The model weights are saved in model.pt and model metadata in model.json

        :arg directory: directory to save in
        """
        assert self.initialised
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.state_dict(), os.path.join(directory, "model.pt"))
        config = dict(dimensions=self.dimensions, architecture=self.architecture)
        with open(os.path.join(directory, "model.json"), "w", encoding="utf8") as f:
            json.dump(config, f, indent=4)

    def load(self, directory):
        """Load model from disk

        :arg directory: directory to load model from
        """
        with open(os.path.join(directory, "model.json"), "r", encoding="utf8") as f:
            config = json.load(f)
        if not self.initialised:
            self.setup(
                config["dimensions"]["n_ref"],
                config["dimensions"]["n_func_in_dynamic"],
                config["dimensions"]["n_func_in_ancillary"],
                config["dimensions"]["n_func_target"],
                config["architecture"],
            )
        self.load_state_dict(
            torch.load(os.path.join(directory, "model.pt"), weights_only=True)
        )
        self.eval()
