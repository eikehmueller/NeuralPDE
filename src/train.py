from timeit import default_timer as timer
from datetime import timedelta

start = timer()

import torch
from torch.utils.data import DataLoader
from output_functions import clear_output
from firedrake import (
    UnitIcosahedralSphereMesh,
    FunctionSpace,
    Function,
    VTKFile,
    errornorm
)

import numpy as np
import argparse

clear_output()

parser = argparse.ArgumentParser()
default_path = "/home/katie795/internship/NeuralPDE/output"
parser.add_argument(
    "--path_to_output_folder",
    type=str,
    action="store",
    default=default_path,
    help="path to output folder",
)

args = parser.parse_args()
path_to_output  = args.path_to_output_folder

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(f"{path_to_output}/tensorboard_logs/solid_body_rotation_experiment_1")

from neural_pde.spherical_patch_covering import SphericalPatchCovering
from neural_pde.patch_encoder import PatchEncoder
from neural_pde.patch_decoder import PatchDecoder
from neural_pde.data_generator import AdvectionDataset
from neural_pde.neural_solver import Katies_NeuralSolver, NeuralSolver

# construct spherical patch covering with
# arg1: number of refinements of the icosahedral sphere
# arg2: number of radial points on each patch
spherical_patch_covering = SphericalPatchCovering(0, 2)

print(f"number of patches               = {spherical_patch_covering.n_patches}")
print(f"patchsize                       = {spherical_patch_covering.patch_size}")
print(f"number of points in all patches = {spherical_patch_covering.n_points}")

num_ref = 2
mesh = UnitIcosahedralSphereMesh(num_ref) # create the mesh
V = FunctionSpace(mesh, "CG", 1) # define the function space
h = 1 / (np.sin(2 * np.pi / 5))# from firedrake documentation

# number of dynamic fields: scalar tracer
n_dynamic = 1
# number of ancillary fields: x-, y- and z-coordinates
n_ancillary = 3
# dimension of latent space
latent_dynamic_dim = 7 # picked to hopefully capture the behaviour wanted
# dimension of ancillary space
latent_ancillary_dim = 3 # also picked to hopefully resolve the behaviour
# number of output fields: scalar tracer
n_output = 1

# encoder models
# dynamic encoder model: map all fields to the latent space
# input:  (n_dynamic+n_ancillary, patch_size)
# output: (latent_dynamic_dim)
dynamic_encoder_model = torch.nn.Sequential(
    torch.nn.Flatten(start_dim=-2, end_dim=-1),
    torch.nn.Linear(
        in_features=(n_dynamic + n_ancillary) * spherical_patch_covering.patch_size, # size of each input sample
        out_features=latent_dynamic_dim, # size of each output sample
    ),
).double() # double means to cast to double precision (float128)

# ancillary encoder model: map ancillary fields to ancillary space
# input:  (n_ancillary, patch_size)
# output: (latent_ancillary_dim)
ancillary_encoder_model = torch.nn.Sequential(
    torch.nn.Flatten(start_dim=-2, end_dim=-1), # since we have 2 inputs, this is the same as flattening at 0
    # and this will lead to a completely flatarray
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
        in_features=4 * (latent_dynamic_dim + latent_ancillary_dim), # do we use a linear model here?? Or do we need a nonlinear part
        out_features=latent_dynamic_dim,
    ),
).double()

# dataset
phi = 0.1 # rotation angle of the data
degree = 4 # degree of the polynomials on the dataset
n_train_samples = 200 # number of samples in the training dataset
n_valid_samples = 32 # needs to be larger than the batch size!!
batchsize = 32 # number of samples to use in each batch

train_ds = AdvectionDataset(V, n_train_samples, phi, degree)
valid_ds = AdvectionDataset(V, n_valid_samples, phi, degree, seed=123456) 

train_dl = DataLoader(train_ds, batch_size=batchsize, shuffle=True, drop_last=True)
valid_dl = DataLoader(valid_ds, batch_size=batchsize , drop_last=True)

assert_testing = {
    "batchsize" : batchsize,
    "n_patches" : spherical_patch_covering.n_patches,
    "d_lat"     : latent_dynamic_dim + latent_ancillary_dim,
    "d_dyn"     : latent_dynamic_dim
}

# visualise the first object in the training dataset 
u_in = Function(V, name="input")
u_in.dat.data[:] = train_ds[0][0][0].numpy()
u_target = Function(V, name="target")
u_target.dat.data[:] = train_ds[0][1].numpy()
file = VTKFile(f"{path_to_output}/training_example.pvd")
file.write(u_in, u_target) # u_target is rotated phi degees CLOCKWISE from u_in

# Full model: encoder + processor + decoder
model = torch.nn.Sequential(
    PatchEncoder(
        V,
        spherical_patch_covering,
        dynamic_encoder_model,
        ancillary_encoder_model,
        n_dynamic,
    ),
    NeuralSolver(spherical_patch_covering, 
                        interaction_model,
                        nsteps=1, 
                        stepsize=1,
                        assert_testing=assert_testing),
    PatchDecoder(V, spherical_patch_covering, decoder_model),
)

opt = torch.optim.Adam(model.parameters(), lr=0.00005) 

## Which Loss function is the best one to use???

def rough_L2_error(y_pred, yb):
    # area of an element in a unit icosahedral mesh
    h_squared = (4 * np.pi) / (20 * (4.0 ** num_ref))
    loss = torch.mean(torch.sum((y_pred - yb)**2  * h_squared))

    return loss

def normalised_L2_error(y_pred, yb):
    # length of an edge in a unit icosahedral mesh
    loss = torch.mean(
        torch.sum(torch.sum((y_pred - yb)**2, dim=1), dim=0) 
        /torch.sum(torch.sum((yb)**2, dim=1), dim=0)
        )
    return loss


def accurate_L2_error(y_pred, yb):
    # DOES NOT WORK YET
    # More accurate, but more expensive L2_error
    # this has no mechanism to calculate backwards gradients
    error = 0
    for _ in range(yb.shape[0]):
        u_ex = Function(V)
        u_ex.dat.data[:] = yb[0].squeeze().detach().numpy()
        u_h = Function(V) # our approximated expression
        u_h.dat.data[:] = y_pred[0].squeeze().detach().numpy()
        error += errornorm(u_ex, u_h)
    return error

        
loss_fn = torch.nn.MSELoss() # mean squared error loss function 
loss_history = []

nepoch = 50

# main training loop
for epoch in range(nepoch):

    # training loop
    model.train(True)
    i = 0 
    for Xb, yb in train_dl:
        opt.zero_grad() # resets all of the gradients to zero, otherwise the gradients are accumulated
        y_pred = model(Xb)

        avg_loss = normalised_L2_error(y_pred, yb)  # only if dataloader drops last
        avg_loss.backward() 
        opt.step() # adjust the parameters by the gradient collected in the backwards pass

        train_x = epoch * len(train_dl) + i + 1
        writer.add_scalar("Loss/train", avg_loss, train_x)
        i += 1 

    running_vloss = 0.0

    # Disable gradient computation and reduce memory consumption.
    # validation loop
    model.eval()
    with torch.no_grad():
        for Xv, yv in valid_dl:
            yv_pred = model(Xv)
            avg_vloss = normalised_L2_error(yv_pred, yv)

    print(f'Epoch {epoch}: Training loss: {avg_loss}, Validation loss: {avg_vloss}')
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training loss' : avg_loss, 'Validation loss' : avg_vloss },
                    epoch + 1)
    writer.flush()
    
    loss_history.append(avg_loss.item())



data_index = 0 # should be between 0 and n_valid_samples
# visualise the first object in the training dataset 
u_in = Function(V, name="input")
u_in.dat.data[:] = valid_ds[0][0][0].numpy()
u_target = Function(V, name="target")
u_target.dat.data[:] = valid_ds[0][1].numpy()
u_predicted_values = model(valid_ds[0][0]).squeeze()
u_predicted = Function(V, name="predicted")
u_predicted.dat.data[:] = u_predicted_values.detach().numpy()
file = VTKFile(f"{path_to_output}/validation_example.pvd")
file.write(u_in, u_target, u_predicted) 

#tensorboard session
#dataiter = iter(train_dl)
#X, y = next(dataiter)
#writer.add_graph(model, X)
#writer.close()

end = timer()
print(f'Runtime: {timedelta(seconds=end-start)}')
