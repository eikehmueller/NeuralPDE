from timeit import default_timer as timer
from datetime import timedelta

start = timer()

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from firedrake import *
from firedrake.adjoint import *
import tqdm
import gc
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
    "--path_to_output_folder",
    type=str,
    action="store",
    help="path to output folder",
    default="output",
)

parser.add_argument(
    "--train_data",
    type=str,
    action="store",
    help="name of file with training dataset",
    default="data/data_train.h5",
)

parser.add_argument(
    "--valid_data",
    type=str,
    action="store",
    help="name of file with validation dataset",
    default="data/data_valid.h5",
)

parser.add_argument(
    "--dual_ref",
    type=int,
    action="store",
    help="number of refinements of dual mesh",
    default=0,
)
parser.add_argument(
    "--n_radial",
    type=int,
    action="store",
    help="number of radial points on each patch",
    default=3,
)
parser.add_argument(
    "--latent_dynamic_dim",
    type=int,
    action="store",
    help="dimension of dynamic latent space",
    default=7,
)
parser.add_argument(
    "--latent_ancillary_dim",
    type=int,
    action="store",
    help="dimension of ancillary latent space",
    default=3,
)
parser.add_argument(
    "--batchsize",
    type=int,
    action="store",
    help="size of batches",
    default=8,
)
parser.add_argument(
    "--nt",
    type=int,
    action="store",
    help="number of timesteps for processor",
    default=4,
)
parser.add_argument(
    "--dt",
    type=float,
    action="store",
    help="size of timestep for processor",
    default=0.25,
)
parser.add_argument(
    "--learning_rate",
    type=float,
    action="store",
    help="learning rate",
    default=0.0006,
)
parser.add_argument(
    "--nepoch", type=int, action="store", help="number of epochs", default=100
)
parser.add_argument(
    "--n_dynamic", type=int, action="store", help="number dynamic fields", default=1
)
args, _ = parser.parse_known_args()
path_to_output = args.path_to_output_folder

# construct spherical patch covering
spherical_patch_covering = SphericalPatchCovering(args.dual_ref, args.n_radial)
print()
print(f"==== data ====")
print()

show_hdf5_header(args.train_data)
print()
show_hdf5_header(args.valid_data)
print()
train_ds = load_hdf5_dataset(args.train_data)
valid_ds = load_hdf5_dataset(args.valid_data)

filename = f"{path_to_output}/hyperparameters.txt"
with open(filename, "w", encoding="utf8") as f:
    print(f"  dual mesh refinements           = {args.dual_ref}", file=f)
    print(f"  radial points per patch         = {args.n_radial}", file=f)
    print(
        f"  points per patch                = {spherical_patch_covering.patch_size}",
        file=f,
    )
    print(
        f"  number of patches               = {spherical_patch_covering.n_patches}",
        file=f,
    )
    print(
        f"  number of points in all patches = {spherical_patch_covering.n_points}",
        file=f,
    )
    print(f"  dynamic latent variables        = {args.latent_dynamic_dim}", file=f)
    print(f"  ancillary latent variables      = {args.latent_ancillary_dim}", file=f)
    print(f"  batchsize                       = {args.batchsize}", file=f)
    print(f"  number of timesteps             = {args.nt}", file=f)
    print(f"  size of timesteps               = {args.dt}", file=f)
    print(f"  learning rate                   = {args.learning_rate}", file=f)
    print(f"  number of epochs                = {args.nepoch}", file=f)

print()
print(f"==== parameters ====")
print()
with open(filename, "r", encoding="utf8") as f:
    print(f.read())


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
        out_features=args.latent_dynamic_dim,  # size of each output sample
    ),
).double()

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
        out_features=args.latent_ancillary_dim,
    ),
).double()

# decoder model: map latent variables to variables on patches
# input:  (d_latent)
# output: (n_out,patch_size)
decoder_model = torch.nn.Sequential(
    torch.nn.Linear(
        in_features=args.latent_dynamic_dim + args.latent_ancillary_dim,
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
).double()

# interaction model: function on latent space
interaction_model = torch.nn.Sequential(
    torch.nn.Flatten(start_dim=-2, end_dim=-1),
    torch.nn.Linear(
        in_features=4 * (args.latent_dynamic_dim + args.latent_ancillary_dim),
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
        out_features=args.latent_dynamic_dim,
    ),
).double()

n_train_samples = train_ds.n_samples
n_valid_samples = valid_ds.n_samples

train_dl = DataLoader(train_ds, batch_size=args.batchsize, shuffle=True, drop_last=True)
valid_dl = DataLoader(valid_ds, batch_size=args.batchsize, drop_last=True)

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
        nsteps=args.nt,
        stepsize=args.dt,
    ),
    PatchDecoder(V, spherical_patch_covering, decoder_model),
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # transfer the model to the GPU

optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

writer = SummaryWriter(flush_secs=5)
# main training loop
for epoch in range(args.nepoch):
    print(f"epoch {epoch + 1} of {args.nepoch}")
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
        train_loss += loss.item() / (n_train_samples // args.batchsize)
        del Xb  # clearing the cache
        del yb
        gc.collect()
        torch.cuda.empty_cache()

    # validation
    model.train(False)
    valid_loss = 0
    for Xv, yv in valid_dl:
        Xv = Xv.to(device)  # move to GPU
        yv = yv.to(device)  # move to GPU
        yv_pred = model(Xv)  # make a prediction
        loss = loss_fn(yv_pred, yv)  # calculate the loss
        valid_loss += loss.item() / (n_valid_samples // args.batchsize)

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
