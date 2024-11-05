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

from output_functions import clear_output, write_to_vtk

from neural_pde.spherical_patch_covering import SphericalPatchCovering
from neural_pde.patch_encoder import PatchEncoder
from neural_pde.patch_decoder import PatchDecoder
from neural_pde.data_classes import load_hdf5_dataset, show_hdf5_header
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

args, _ = parser.parse_known_args()
path_to_output = args.path_to_output_folder

clear_output(path_to_output)  # clear the output folder of previous works

##### HYPERPARAMETERS #####
test_number = "39"
dual_ref = 0  # refinement of the dual mesh
n_radial = 3  # number of radial points on each patch
latent_dynamic_dim = 7  # dimension of dynamic latent space
latent_ancillary_dim = 3  # dimension of ancillary latent space
degree = 4  # degree of the polynomials on the dataset
batchsize = 8  # number of samples to use in each batch
nt = 4  # number of timesteps
dt = 0.25  # size of the timesteps
lr = 0.0006  # learning rate of the optimizer
nepoch = 100  # number of epochs
n_dynamic = 1  # number of dynamic fields: scalar tracer
n_ancillary = 3  # number of ancillary fields: x-, y- and z-coordinates
n_output = 1  # number of output fields: scalar tracer
##### HYPERPARAMETERS #####

# construct spherical patch covering
spherical_patch_covering = SphericalPatchCovering(dual_ref, n_radial)
print("")
print(f"running test number {test_number}")
print("")

show_hdf5_header(args.train_data)
show_hdf5_header(args.valid_data)

train_ds = load_hdf5_dataset(args.train_data)
valid_ds = load_hdf5_dataset(args.valid_data)

filename = f"{path_to_output}/hyperparameters_test{test_number}.txt"
with open(filename, "w", encoding="utf8") as f:
    print(f"dual mesh refinements           = {dual_ref}", file=f)
    print(f"radial points per patch         = {n_radial}", file=f)
    print(
        f"total points per patch          = {spherical_patch_covering.patch_size}",
        file=f,
    )
    print(
        f"number of patches               = {spherical_patch_covering.n_patches}",
        file=f,
    )
    print(
        f"number of points in all patches = {spherical_patch_covering.n_points}", file=f
    )
    print(f"dynamic latent variables        = {latent_dynamic_dim}", file=f)
    print(f"ancillary latent variables      = {latent_ancillary_dim}", file=f)
    print(f"degree of polynomial in dataset = {degree}", file=f)
    print(f"batchsize                       = {batchsize}", file=f)
    print(f"number of timesteps             = {nt}", file=f)
    print(f"size of timesteps               = {dt}", file=f)
    print(f"learning rate                   = {lr}", file=f)
    print(f"number of epochs                = {nepoch}", file=f)

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
        in_features=(n_dynamic + n_ancillary)
        * spherical_patch_covering.patch_size,  # size of each input sample
        out_features=latent_dynamic_dim,  # size of each output sample
    ),
).double()

# ancillary encoder model: map ancillary fields to ancillary space
# input:  (n_ancillary, patch_size)
# output: (latent_ancillary_dim)
ancillary_encoder_model = torch.nn.Sequential(
    torch.nn.Flatten(start_dim=-2, end_dim=-1),
    torch.nn.Linear(
        in_features=n_ancillary * spherical_patch_covering.patch_size,
        out_features=latent_ancillary_dim,
    ),
).double()

# decoder model: map latent variables to variables on patches
# input:  (d_latent)
# output: (n_out,patch_size)
decoder_model = torch.nn.Sequential(
    torch.nn.Linear(
        in_features=latent_dynamic_dim + latent_ancillary_dim,
        out_features=n_output * spherical_patch_covering.patch_size,
    ),
    torch.nn.Unflatten(
        dim=-1, unflattened_size=(n_output, spherical_patch_covering.patch_size)
    ),
).double()

# interaction model: function on latent space
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

n_train_samples = train_ds.n_samples
n_valid_samples = valid_ds.n_samples

train_dl = DataLoader(train_ds, batch_size=batchsize, shuffle=True, drop_last=True)
valid_dl = DataLoader(valid_ds, batch_size=batchsize, drop_last=True)

# assert testing can be included as an extra test in the processor, to check that the tensors have the correct size.
assert_testing = {
    "batchsize": batchsize,
    "n_patches": spherical_patch_covering.n_patches,
    "d_lat": latent_dynamic_dim + latent_ancillary_dim,
    "d_dyn": latent_dynamic_dim,
}

# Full model: encoder + processor + decoder
model = torch.nn.Sequential(
    PatchEncoder(
        V,
        spherical_patch_covering,
        dynamic_encoder_model,
        ancillary_encoder_model,
        n_dynamic,
    ),
    NeuralSolver(
        spherical_patch_covering,
        interaction_model,
        nsteps=nt,
        stepsize=dt,
    ),
    PatchDecoder(V, spherical_patch_covering, decoder_model),
)


device = "cpu"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # transfer the model to the GPU

optimiser = torch.optim.Adam(model.parameters(), lr=lr)

writer = SummaryWriter(flush_secs=5)
# main training loop
for epoch in range(nepoch):
    print(f"epoch {epoch + 1} of {nepoch}")
    train_loss = 0
    for Xb, yb in tqdm.tqdm(train_dl):
        Xb = Xb.to(device)  # move to GPU
        yb = yb.to(device)  # move to GPU
        y_pred = model(Xb)  # make a prediction
        optimiser.zero_grad()  # resets all of the gradients to zero, otherwise the gradients are accumulated
        loss = loss_fn(y_pred, yb)  # calculate the loss
        loss.backward()  # take the backwards gradient
        optimiser.step()  # adjust the parameters by the gradient collected in the backwards pass
        # data collection for the model
        train_loss += loss.item() / (n_train_samples // batchsize)
        del Xb  # clearing the cache
        del yb
        gc.collect()
        torch.cuda.empty_cache()

    # validation loop
    valid_loss = 0
    for Xv, yv in valid_dl:
        Xv = Xv.to(device)  # move to GPU
        yv = yv.to(device)  # move to GPU
        yv_pred = model(Xv)  # make a prediction
        loss = loss_fn(yv_pred, yv)  # calculate the loss
        valid_loss += loss.item() / (n_valid_samples // batchsize)

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

# visualise the first object in the training dataset
host_model = model.cpu()
write_to_vtk(
    V, name="input", dof_values=valid_ds[1][0][0].numpy(), path_to_output=path_to_output
)
write_to_vtk(
    V,
    name="target",
    dof_values=valid_ds[0][1].squeeze().cpu().numpy(),
    path_to_output=path_to_output,
)
write_to_vtk(
    V,
    name="predicted",
    dof_values=host_model(valid_ds[0][0]).squeeze().detach().numpy(),
    path_to_output=path_to_output,
)
