from timeit import default_timer as timer
from datetime import timedelta
start = timer()

import torch
from torch.utils.data import DataLoader
from firedrake import *
from firedrake.adjoint import *
import matplotlib.pyplot as plt
import numpy as np
import os
import gc
import argparse

from output_functions import clear_output, write_to_vtk, training_plots
from loss_functions import normalised_L2_error as loss

# Create argparse arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "--path_to_output_folder",
    type=str,
    action="store",
    help="path to output folder",
    required=True
)

args, _ = parser.parse_known_args()
path_to_output  = args.path_to_output_folder

clear_output(path_to_output) # clear the output folder of previous works

##### HYPERPARAMETERS #####
test_number = "1"
dual_ref = 0             # refinement of the dual mesh
n_radial = 2             # number of radial points on each patch
n_ref = 2                # number of refinements of the icosahedral mesh
latent_dynamic_dim = 7   # dimension of dynamic latent space
latent_ancillary_dim = 3 # dimension of ancillary latent space
phi = 0.7854             # approx pi/4
degree = 4               # degree of the polynomials on the dataset
n_train_samples = 512    # number of samples in the training dataset
n_valid_samples = 32     # needs to be larger than the batch size!!
batchsize = 32           # number of samples to use in each batch
accum = 1                # gradient accumulation for larger batchsizes - ASSERT TESTING DOES NOT WORK IF ACCUM /= 1
nt = 4                   # number of timesteps
dt = 0.25                # size of the timesteps
lr = 0.0006              # learning rate of the optimizer
nepoch = 10            # number of epochs
##### HYPERPARAMETERS #####

from neural_pde.spherical_patch_covering import SphericalPatchCovering
from neural_pde.patch_encoder import PatchEncoder
from neural_pde.patch_decoder import PatchDecoder
from neural_pde.data_classes import AdvectionDataset
from neural_pde.neural_solver import NeuralSolver

# construct spherical patch covering with
# arg1: number of refinements of the icosahedral sphere
# arg2: number of radial points on each patch
spherical_patch_covering = SphericalPatchCovering(dual_ref, n_radial)
print('')
print(f"running test number               {test_number}")
print('')

f = open(f"{path_to_output}/hyperparameters.txt", "w")
f.write(f'dual mesh refinements           = {dual_ref}\n')
f.write(f'radial points per patch         = {n_radial}\n')
f.write(f"total points per patch          = {spherical_patch_covering.patch_size}\n")
f.write(f"number of patches               = {spherical_patch_covering.n_patches}\n")
f.write(f"number of points in all patches = {spherical_patch_covering.n_points}\n")
f.write(f'n_ref of original mesh          = {n_ref}\n')
f.write(f'dynamic latent variables        = {latent_dynamic_dim}\n')
f.write(f'ancillary latent variables      = {latent_ancillary_dim}\n')
f.write(f'solid body rotation angle (rad) = {phi}\n')
f.write(f'degree of polynomial in dataset = {degree}\n')
f.write(f'training samples                = {n_train_samples}\n')
f.write(f'validation samples              = {n_valid_samples}\n')
f.write(f'batchsize                       = {batchsize}\n')
f.write(f'gradient accumulation number    = {accum}\n')
f.write(f'number of timesteps             = {nt}\n')
f.write(f'size of timesteps               = {dt}\n')
f.write(f'learning rate                   = {lr}\n')
f.write(f'number of epochs                = {nepoch}\n')

f = open(f"{path_to_output}/hyperparameters.txt", "r")
print(f.read())



mesh = UnitIcosahedralSphereMesh(n_ref) # create the mesh
V = FunctionSpace(mesh, "CG", 1) # define the function space

n_dynamic = 1   # number of dynamic fields: scalar tracer
n_ancillary = 3 # number of ancillary fields: x-, y- and z-coordinates
n_output = 1    # number of output fields: scalar tracer

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
        in_features=4 * (latent_dynamic_dim + latent_ancillary_dim), # do we use a linear model here?? Or do we need a nonlinear part
        out_features=latent_dynamic_dim,
    ),
).double()

train_data = f"data_{n_train_samples}_{n_ref}_{phi}.npy" # name of the training data file
valid_data = f"data_{n_valid_samples}_{n_ref}_{phi}.npy" # name of the validation data file

if not os.path.exists(f"data/{train_data}"):
    print("Data requested does not exist, run data_producer.py to generate the data")
if not os.path.exists(f"data/{valid_data}"):
    print("Data requested does not exist, run data_producer.py to generate the data")

train_ds = AdvectionDataset(V, n_train_samples, phi, degree) 
valid_ds = AdvectionDataset(V, n_valid_samples, phi, degree)
train_ds.load(f"data/{train_data}") # load the data
valid_ds.load(f"data/{valid_data}") # load the data

train_dl = DataLoader(train_ds, batch_size=batchsize//accum, shuffle=True, drop_last=True) 
valid_dl = DataLoader(valid_ds, batch_size=batchsize , drop_last=True)

# assert testing can be included as an extra test in the processor, to check that the tensors have the correct size.
assert_testing = { 
    "batchsize" : batchsize,
    "n_patches" : spherical_patch_covering.n_patches,
    "d_lat"     : latent_dynamic_dim + latent_ancillary_dim,
    "d_dyn"     : latent_dynamic_dim
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
    NeuralSolver(spherical_patch_covering, 
                        interaction_model,
                        nsteps=nt, 
                        stepsize=dt,
                        assert_testing=None),
    PatchDecoder(V, spherical_patch_covering, decoder_model),
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
model = model.to(device) # transfer the model to the GPU

opt = torch.optim.Adam(model.parameters(), lr=lr) 

training_loss = []
training_loss_per_epoch = []
validation_loss_per_epoch = []

# main training loop
for epoch in range(nepoch):
    model.train(True)
    i = 0 
    count = 0  # track count of items since last weight update
    for Xb, yb in train_dl:
        count += len(Xb)
        Xb = Xb.to(device) # move to GPU
        yb = yb.to(device) # move to GPU
        y_pred = model(Xb) # make a prediction
        avg_loss = loss(y_pred, yb)  # calculate the loss
        avg_loss.backward() # take the backwards gradient 

        if count >= batchsize:  # count is greater than accumulation target, so we do a weight update
            opt.step()          # adjust the parameters by the gradient collected in the backwards pass
            opt.zero_grad()     # resets all of the gradients to zero, otherwise the gradients are accumulated
            count = 0
        
        # data collection for the model
        training_loss.append(avg_loss.cpu().detach().numpy())

        del Xb # clearing the cache
        del yb
        gc.collect()
        torch.cuda.empty_cache()
    
    training_loss_per_epoch.append(avg_loss.cpu().detach().numpy())

    # validation loop
    model.eval()
    with torch.no_grad():
        for Xv, yv in valid_dl:
            Xv = Xv.to(device) # move to GPU
            yv = yv.to(device) # move to GPU
            yv_pred = model(Xv) # make a prediction
            avg_vloss = loss(yv_pred, yv) # calculate the loss

    print(f'Epoch {epoch + 1}: Training loss: {avg_loss}, Validation loss: {avg_vloss}')
    validation_loss_per_epoch.append(avg_vloss.cpu().detach().numpy())

end = timer()
print(f'Runtime: {timedelta(seconds=end-start)}')

# visualise the first object in the training dataset 
host_model = model.cpu()
write_to_vtk(V, name="input",     dof_values=valid_ds[1][0][0].numpy(),                             path_to_output=path_to_output)
write_to_vtk(V, name="target",    dof_values=valid_ds[0][1].squeeze().cpu().numpy(),                path_to_output=path_to_output)
write_to_vtk(V, name="predicted", dof_values=host_model(valid_ds[0][0]).squeeze().detach().numpy(), path_to_output=path_to_output)

training_plots(training_loss, training_loss_per_epoch, validation_loss_per_epoch, path_to_output, test_number)