from timeit import default_timer as timer
from datetime import timedelta

start = timer()

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from firedrake import *
from firedrake.adjoint import *
import tqdm
import tomllib
import argparse

from neural_pde.spherical_patch_covering import SphericalPatchCovering
from neural_pde.patch_encoder import PatchEncoder
from neural_pde.patch_decoder import PatchDecoder
from neural_pde.datasets import load_hdf5_dataset, show_hdf5_header
from neural_pde.neural_solver import NeuralSolver
from neural_pde.loss_functions import normalised_mse as loss_fn

# Create argparse arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    type=str,
    action="store",
    help="name of parameter file",
    default="config.toml",
)

parser.add_argument(
    "--path_to_output_folder",
    type=str,
    action="store",
    help="path to output folder",
    default="output",
)

args, _ = parser.parse_known_args()
path_to_output = args.path_to_output_folder

with open(args.config, "rb") as f:
    config = tomllib.load(f)
print()
print(f"==== parameters ====")
print()
with open(args.config, "r") as f:
    for line in f.readlines():
        print(line.strip())

# construct spherical patch covering
spherical_patch_covering = SphericalPatchCovering(
    config["architecture"]["dual_ref"], config["architecture"]["n_radial"]
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
print()
print(f"==== data ====")
print()

show_hdf5_header(config["data"]["train"])
print()
show_hdf5_header(config["data"]["valid"])
print()
train_ds = load_hdf5_dataset(config["data"]["train"])
valid_ds = load_hdf5_dataset(config["data"]["valid"])

mesh = UnitIcosahedralSphereMesh(train_ds.n_ref)  # create the mesh
V = FunctionSpace(mesh, "CG", 1)  # define the function space

# encoder models
# dynamic encoder model: map all fields to the latent space
# input:  (n_dynamic+n_ancillary, patch_size)
# output: (latent_dynamic_dim)
dynamic_encoder_model = torch.nn.Sequential(
    torch.nn.Flatten(start_dim=-2, end_dim=-1),
    torch.nn.Linear(
        in_features=(train_ds.n_func_in_dynamic + train_ds.n_func_in_ancillary)
        * spherical_patch_covering.patch_size,  # size of each input sample
        out_features=16,
    ),
    torch.nn.Softplus(),
    torch.nn.Linear(in_features=16, out_features=16),
    torch.nn.Softplus(),
    torch.nn.Linear(
        in_features=16,
        out_features=config["architecture"]["latent_dynamic_dim"],
    ),
)

# ancillary encoder model: map ancillary fields to ancillary space
# input:  (n_ancillary, patch_size)
# output: (latent_ancillary_dim)
ancillary_encoder_model = torch.nn.Sequential(
    torch.nn.Flatten(start_dim=-2, end_dim=-1),
    torch.nn.Linear(
        in_features=train_ds.n_func_in_ancillary * spherical_patch_covering.patch_size,
        out_features=16,
    ),
    torch.nn.Softplus(),
    torch.nn.Linear(in_features=16, out_features=16),
    torch.nn.Softplus(),
    torch.nn.Linear(
        in_features=16,
        out_features=config["architecture"]["latent_ancillary_dim"],
    ),
)

# decoder model: map latent variables to variables on patches
# input:  (d_latent)
# output: (n_out,patch_size)
decoder_model = torch.nn.Sequential(
    torch.nn.Linear(
        in_features=config["architecture"]["latent_dynamic_dim"]
        + config["architecture"]["latent_ancillary_dim"],
        out_features=16,
    ),
    torch.nn.Softplus(),
    torch.nn.Linear(in_features=16, out_features=16),
    torch.nn.Softplus(),
    torch.nn.Linear(
        in_features=16,
        out_features=train_ds.n_func_target * spherical_patch_covering.patch_size,
    ),
    torch.nn.Unflatten(
        dim=-1,
        unflattened_size=(train_ds.n_func_target, spherical_patch_covering.patch_size),
    ),
)

# interaction model: function on latent space
interaction_model = torch.nn.Sequential(
    torch.nn.Flatten(start_dim=-2, end_dim=-1),
    torch.nn.Linear(
        in_features=4
        * (
            config["architecture"]["latent_dynamic_dim"]
            + config["architecture"]["latent_ancillary_dim"]
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
        out_features=config["architecture"]["latent_dynamic_dim"],
    ),
)

n_train_samples = train_ds.n_samples
n_valid_samples = valid_ds.n_samples

train_dl = DataLoader(
    train_ds, batch_size=config["optimiser"]["batchsize"], shuffle=True, drop_last=True
)
valid_dl = DataLoader(
    valid_ds, batch_size=config["optimiser"]["batchsize"], drop_last=True
)

# Full model: encoder + processor + decoder
model = torch.nn.Sequential(
    PatchEncoder(
        V,
        spherical_patch_covering,
        dynamic_encoder_model,
        ancillary_encoder_model,
        train_ds.n_func_in_dynamic,
    ),
    NeuralSolver(
        spherical_patch_covering,
        interaction_model,
        nsteps=config["processor"]["nt"],
        stepsize=config["processor"]["dt"],
    ),
    PatchDecoder(V, spherical_patch_covering, decoder_model),
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # transfer the model to the GPU
print(f"Running on device {device}")

optimiser = torch.optim.Adam(
    model.parameters(), lr=config["optimiser"]["learning_rate"]
)

writer = SummaryWriter(flush_secs=5)
# main training loop
for epoch in range(config["optimiser"]["nepoch"]):
    print(f"epoch {epoch + 1} of {config["optimiser"]["nepoch"]}")
    train_loss = 0
    model.train(True)
    for Xb, yb in tqdm.tqdm(train_dl):
        Xb = Xb.to(device)  # move to GPU
        yb = yb.to(device)  # move to GPU
        y_pred = model(Xb)  # make a prediction
        optimiser.zero_grad()  # resets all of the gradients to zero, otherwise the gradients are accumulated
        loss = loss_fn(y_pred, yb)  # calculate the loss
        loss.backward()  # take the backwards gradient
        optimiser.step()  # adjust the parameters by the gradient collected in the backwards pass
        # data collection for the model
        train_loss += loss.item() / (
            n_train_samples // config["optimiser"]["batchsize"]
        )

    # validation
    model.train(False)
    valid_loss = 0
    for Xv, yv in valid_dl:
        Xv = Xv.to(device)  # move to GPU
        yv = yv.to(device)  # move to GPU
        yv_pred = model(Xv)  # make a prediction
        loss = loss_fn(yv_pred, yv)  # calculate the loss
        valid_loss += loss.item() / (
            n_valid_samples // config["optimiser"]["batchsize"]
        )

    print(f"    training loss: {train_loss:8.3e}, validation loss: {valid_loss:8.3e}")
    writer.add_scalars(
        "loss",
        {"train": train_loss, "valid": valid_loss},
        epoch,
    )
    print()
writer.flush()

end = timer()
print(f"Runtime: {timedelta(seconds=end-start)}")

# visualise the validation dataset
host_model = model.cpu()

for j, (X, y_target) in enumerate(iter(valid_ds)):
    y_pred = host_model(X)

    f_input = Function(V, name="input")
    f_input.dat.data[:] = X.detach().numpy()[0, :]

    f_target = Function(V, name="target")
    f_target.dat.data[:] = y_target.detach().numpy()[0, :]

    f_pred = Function(V, name="pedicted")
    f_pred.dat.data[:] = y_pred.detach().numpy()[0, :]

    file = VTKFile(f"{path_to_output}/output_{j:04d}.pvd")
    file.write(f_input, f_target, f_pred)
