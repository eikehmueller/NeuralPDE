"""Encoder-processor-decoder model"""

from firedrake import *
import torch
import os
import json

from neural_pde.grid.patch_encoder import PatchEncoder
from neural_pde.grid.patch_decoder import PatchDecoder
from neural_pde.grid.decoder import Decoder
from neural_pde.grid.spherical_patch_covering import SphericalPatchCovering
from neural_pde.solver.neural_solver import (
    ForwardEulerNeuralSolver,
    SymplecticNeuralSolver,
)


__all__ = ["build_model", "load_model"]


def build_model(
    n_ref,
    n_func_in_dynamic,
    n_func_in_ancillary,
    n_func_target,
    architecture,
    mean=0,
    std=1,
):
    """
    Construct encoder - processor - decoder model

    :arg n_ref: number of refinement steps of icosahedral mesh
    :arg n_func_in_dynamic: number of dynamic input functions
    :arg n_func_in_ancillary: number of ancillary input functions
    :arg n_func_in_target: number of output functions
    :arg architecture: dictionary that describes network architecture
    :arg mean: mean of the batchsize over each function at every point
    :arg std: standard devation of the batchsize over each function at every point
    """
    model = NeuralPDEModel()
    model.setup(
        n_ref,
        n_func_in_dynamic,
        n_func_in_ancillary,
        n_func_target,
        architecture,
        mean,
        std,
    )
    return model


def load_model(directory):
    """Load model from disk

    :arg directory: directory containing the saved model"""
    checkpoint = torch.load(os.path.join(directory, "checkpoint.pt"))

    model = NeuralPDEModel()
    model.load(directory, checkpoint)

    optimiser = torch.optim.Adam(model.parameters())
    optimiser.load_state_dict(checkpoint["optimizer"])

    epoch = checkpoint["epoch"]

    return model, optimiser, epoch


class NeuralPDEModel(torch.nn.Module):
    """Class representing the encoder - processor - decoder network"""

    def __init__(self):
        """Initialise a new instance with empty model

        :arg mean: mean of the batchsize over each function at every point
        :arg std: standard devation of the batchsize over each function at every point
        """
        super().__init__()
        self.architecture = None
        self.dimensions = None
        self.initialised = False

    def setup(
        self,
        n_ref,
        n_func_in_dynamic,
        n_func_in_ancillary,
        n_func_target,
        architecture,
        mean=0,
        std=1,
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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.architecture = architecture
        self.dimensions = dict(
            n_ref=n_ref,
            n_func_in_dynamic=n_func_in_dynamic,
            n_func_in_ancillary=n_func_in_ancillary,
            n_func_target=n_func_target,
        )
        self.n_func_in_dynamic = n_func_in_dynamic
        self.mean = mean
        self.std = std


        self.x_mean = self.mean[:n_func_in_dynamic, :].unsqueeze(0).to(torch.float32).to(device)


        self.x_std = self.std[:n_func_in_dynamic, :].unsqueeze(0).to(torch.float32).to(device)


        self.w_mean = (
            self.mean[n_func_in_dynamic + n_func_in_ancillary :, :]
            .unsqueeze(0)
            .to(torch.float32)
        ).to(device)

        self.w_std = (
            self.std[n_func_in_dynamic + n_func_in_ancillary :, :]
            .unsqueeze(0)
            .to(torch.float32)
        ).to(device)

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
        print(
            f"  total number of unknowns             = {V.dof_count * spherical_patch_covering.n_patches}"
        )

        # encoder models
        # dynamic encoder model: map all fields to the latent space
        # input:  (n_dynamic+n_ancillary, patch_size)
        # output: (latent_dynamic_dim)
        n_hidden = 32
        dynamic_encoder_model = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=-2, end_dim=-1),
            torch.nn.Linear(
                in_features=(n_func_in_dynamic + n_func_in_ancillary)
                * spherical_patch_covering.patch_size,  # size of each input sample
                out_features=n_hidden,
            ),
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
            torch.nn.Linear(
                in_features=n_hidden,
                out_features=architecture["latent_ancillary_dim"],
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
        if architecture["neural_solver"] == "symplectic":
            n_hidden_hamiltonian = 32
            # local Hamiltonians
            H_q_local = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=architecture["latent_dynamic_dim"]
                    + 2 * architecture["latent_ancillary_dim"],
                    out_features=n_hidden_hamiltonian,
                ),
                torch.nn.Softplus(),
                torch.nn.Linear(
                    in_features=n_hidden_hamiltonian,
                    out_features=1,
                ),
                torch.nn.Softplus(),
            )
            H_p_local = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=architecture["latent_dynamic_dim"]
                    + 2 * architecture["latent_ancillary_dim"],
                    out_features=n_hidden_hamiltonian,
                ),
                torch.nn.Softplus(),
                torch.nn.Linear(
                    in_features=n_hidden_hamiltonian,
                    out_features=1,
                ),
                torch.nn.Softplus(),
            )
            self.add_module(
                "NeuralSolver",
                SymplecticNeuralSolver(
                    spherical_patch_covering.dual_mesh,
                    architecture["latent_dynamic_dim"],
                    architecture["latent_ancillary_dim"],
                    H_q_local,
                    H_p_local,
                    stepsize=architecture["dt"],
                ),
            )
        elif architecture["neural_solver"] == "forward_euler":
            n_hidden_interaction = 32
            # interaction model: function on latent space
            interaction_model = torch.nn.Sequential(
                torch.nn.Flatten(start_dim=-2, end_dim=-1),
                torch.nn.Linear(
                    in_features=4
                    * (
                        architecture["latent_dynamic_dim"]
                        + architecture["latent_ancillary_dim"]
                    ),
                    out_features=n_hidden_interaction,
                ),
                torch.nn.Softplus(),
                torch.nn.Linear(
                    in_features=n_hidden_interaction,
                    out_features=architecture["latent_dynamic_dim"],
                ),
            )
            self.add_module(
                "NeuralSolver",
                ForwardEulerNeuralSolver(
                    spherical_patch_covering.dual_mesh,
                    interaction_model,
                    stepsize=architecture["dt"],
                ),
            )
        else:
            raise RuntimeError("Unknown neural solver: ", architecture["neural_solver"])
        if architecture["decoder"] == "patch":
            # decoder model: map latent variables to variables on patches
            # input:  (nu*d_latent+n_ancil)
            # output: (n_out)
            decoder_model = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=architecture["latent_dynamic_dim"]
                    + architecture["latent_ancillary_dim"],
                    out_features=n_hidden,
                ),
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
            self.add_module(
                "PatchDecoder",
                PatchDecoder(
                    V,
                    spherical_patch_covering,
                    decoder_model,
                ),
            )
        elif architecture["decoder"] == "nearest_neighbour":
            decoder_model = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=architecture["nu"]
                    * (
                        architecture["latent_dynamic_dim"]
                        + architecture["latent_ancillary_dim"]
                    )
                    + n_func_in_ancillary,
                    out_features=n_hidden,
                ),
                torch.nn.Softplus(),
                torch.nn.Linear(
                    in_features=n_hidden,
                    out_features=n_func_target,
                ),
            )

            self.add_module(
                "Decoder",
                Decoder(
                    V,
                    spherical_patch_covering.dual_mesh,
                    decoder_model,
                    nu=architecture["nu"],
                ),
            )

        self.initialised = True

    def normalise_data(self, x):
        x_mean = self.x_mean.to(x.device)
        print(f"X_mean is {torch.max(x_mean)}")
        x_std = self.x_std.to(x.device)
        x_normalised = x.clone()
        x_normalised[:, : self.n_func_in_dynamic, :] = (
            x[:, : self.n_func_in_dynamic, :] - x_mean
        ) / x_std
        print(f"xnorm is on device {x_normalised.device}")
        print(f"Max of xnorm is {torch.max(x_normalised)}")
        print(f"Difference between x and x_normalised is {torch.max(x_normalised - x)}")
        return x_normalised

    def forward(self, x, t_final):
        """Forward pass of the model
        :arg x: input tensor of shape (batch_size, n_func_in_dynamic + n_func_in_ancillary, n_vertex)
        :arg t_final: final time for each sample, tensor of shape (batch_size,)
        """
        x_normalised = self.normalise_data(x)

        y = self.PatchEncoder(x_normalised)
        z = self.NeuralSolver(y, t_final)

        if hasattr(self, "PatchDecoder"):
            w = self.PatchDecoder(z)
        elif hasattr(self, "Decoder"):
            x_ancil = x[:, self.dimensions["n_func_in_dynamic"] :, :]
            w = self.Decoder(z, x_ancil)
        else:
            raise RuntimeError("Model has no decoder attribute!")
        # w2 = x_normalised[:, :self.n_func_in_dynamic, :]
        w_final = w * self.w_std + self.w_mean
        # w2_final = w2 * self.w_std + self.w_mean

        # print(f'Error is {torch.mean(x[:, :self.n_func_in_dynamic, :] - w2_final)}')
        return w_final

    def save(self, state, directory):
        """Save model to disk

        The model weights are saved in model.pt and model metadata in model.json

        :arg state: dictionary with state of model, optimiser and epoch
        :arg directory: directory to save in
        """
        assert self.initialised
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(state, os.path.join(directory, "checkpoint.pt"))

        config = dict(
            dimensions=self.dimensions,
            architecture=self.architecture,
            mean=self.mean.numpy().tolist(),
            std=self.std.numpy().tolist(),
        )
        with open(
            os.path.join(directory, "checkpoint.json"), "w", encoding="utf8"
        ) as f:
            json.dump(config, f, indent=4)

    def load(self, directory, checkpoint):
        """Load model from disk

        :arg directory: directory to load model from
        :arg checkpoint: dictionary with state of the model
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with open(
            os.path.join(directory, "checkpoint.json"), "r", encoding="utf8"
        ) as f:
            config = json.load(f)

        tensor_mean = torch.FloatTensor(config["mean"]).to(device)
        tensor_std = torch.FloatTensor(config["std"]).to(device)

        if not self.initialised:
            self.setup(
                config["dimensions"]["n_ref"],
                config["dimensions"]["n_func_in_dynamic"],
                config["dimensions"]["n_func_in_ancillary"],
                config["dimensions"]["n_func_target"],
                config["architecture"],
                tensor_mean,
                tensor_std,
            )
        self.load_state_dict(checkpoint["state_dict"])

        self.eval()
