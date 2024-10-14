from timeit import default_timer as timer
from datetime import timedelta
start = timer()

import torch
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from firedrake import *
from firedrake.adjoint import *
import matplotlib.pyplot as plt
import numpy as np
import os
import gc

from output_functions import clear_output, write_to_vtk
from loss_functions import normalised_L2_error as loss

import argparse

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

clear_output(path_to_output)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using the device {device}')

##### HYPERPARAMETERS #####
test_number = "45"
dual_ref = 0             # refinement of the dual mesh
n_radial = 3             # number of radial points on each patch
n_ref = 2                # number of refinements of the icosahedral mesh
latent_dynamic_dim = 14   # dimension of dynamic latent space
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
nepoch = 100            # number of epochs
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
print(f"Running test {test_number}")
print(f"number of patches               = {spherical_patch_covering.n_patches}")
print(f"patchsize                       = {spherical_patch_covering.patch_size}")
print(f"number of points in all patches = {spherical_patch_covering.n_points}")

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

train_data = f"data_{n_train_samples}_{n_ref}_{phi}.npy"
valid_data = f"data_{n_valid_samples}_{n_ref}_{phi}.npy"

if not os.path.exists(f"data/{train_data}"):
    print("Data requested does not exist, run data_producer.py to generate the data")
if not os.path.exists(f"data/{valid_data}"):
    print("Data requested does not exist, run data_producer.py to generate the data")

train_ds = AdvectionDataset(V, n_train_samples, phi, degree)
valid_ds = AdvectionDataset(V, n_valid_samples, phi, degree)
train_ds.load(f"data/{train_data}")
valid_ds.load(f"data/{valid_data}")

train_dl = DataLoader(train_ds, batch_size=batchsize//accum, shuffle=True, drop_last=True)
valid_dl = DataLoader(valid_ds, batch_size=batchsize , drop_last=True)

assert_testing = {
    "batchsize" : batchsize,
    "n_patches" : spherical_patch_covering.n_patches,
    "d_lat"     : latent_dynamic_dim + latent_ancillary_dim,
    "d_dyn"     : latent_dynamic_dim
}

# visualise the first object in the training dataset 
#write_to_vtk(V, name="input_training", dof_values=train_ds[0][0][0].numpy(), path_to_output=path_to_output)
#write_to_vtk(V, name="target_training", dof_values=train_ds[0][1].numpy(), path_to_output=path_to_output)

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
model = model.to(device)

opt = torch.optim.Adam(model.parameters(), lr=lr) 

training_loss = []
training_loss_per_epoch = []
validation_loss_per_epoch = []


# Profiling the code
'''
train_example = torch.randn(4, V.dim()).double().to(device)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(train_example)
print('Data for execution time')
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

with profile(activities=[ProfilerActivity.CUDA],
        profile_memory=True, record_shapes=True) as prof:
    model(train_example)

print('Data for memory usage')
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))


with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(train_example)

prof.export_chrome_trace("trace.json")

print('Data for stack tracing')
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True,
) as prof:
    model(train_example)


# Print aggregated stats
print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=2))

'''
time = 0
print('Starting training loop')
# main training loop
for epoch in range(nepoch):
    model.train(True)
    i = 0 
    count = 0  # track count of items since last weight update
    for Xb, yb in train_dl:
        count += len(Xb)
        Xb = Xb.to(device)
        yb = yb.to(device)
        y_pred = model(Xb)
        avg_loss = loss(y_pred, yb)  # only if dataloader drops last
        avg_loss.backward() 

        if count >= batchsize:  # count is greater than accumulation target, so we do a weight update
            opt.step()          # adjust the parameters by the gradient collected in the backwards pass
            opt.zero_grad()     # resets all of the gradients to zero, otherwise the gradients are accumulated
            count = 0
        
        # data collection for the model
        zero_loss = loss(torch.zeros_like(y_pred), y_pred)
        train_x = epoch * len(train_dl) + i + 1
        training_loss.append(avg_loss.cpu().detach().numpy())
        i += 1 

        del Xb # clearing the cache
        del yb
        gc.collect()
        torch.cuda.empty_cache()
    
    training_loss_per_epoch.append(avg_loss.cpu().detach().numpy())

    # validation loop
    model.eval()
    with torch.no_grad():
        for Xv, yv in valid_dl:
            Xv = Xv.to(device)
            yv = yv.to(device)
            yv_pred = model(Xv)

            avg_vloss = loss(yv_pred, yv)
    
    if time % 5 == 0:
        print(f'Time is {time//5}')
        write_to_vtk(V, name="vtk_animation", dof_values=yv_pred[1][0].cpu().numpy(), path_to_output=path_to_output, time=time//5)

    print(f'Epoch {epoch + 1}: Training loss: {avg_loss}, Validation loss: {avg_vloss}')
    validation_loss_per_epoch.append(avg_vloss.cpu().detach().numpy())
    time += 1


# visualise the first object in the training dataset 
host_model = model.cpu()

#write_to_vtk(V, name="input_validation", dof_values=valid_ds[1][0][0].numpy(), path_to_output=path_to_output)
write_to_vtk(V, name="target_validation", dof_values=valid_ds[1][1].numpy(), path_to_output=path_to_output)
#write_to_vtk(V, name="predicted_validation", dof_values=host_model(valid_ds[1][0]).squeeze().detach().numpy(), path_to_output=path_to_output)

end = timer()
print(f'Runtime: {timedelta(seconds=end-start)}')

training_iterations = np.arange(0.0, len(training_loss), 1)
epoch_iterations = np.arange(0.0, len(training_loss_per_epoch), 1)

fig1, ax1 = plt.subplots()
ax1.set_ylim([0, 1.1])
ax1.plot(training_iterations, np.array(training_loss))
ax1.set(xlabel='Number of training iterations', ylabel=r'Normalized $L^2$ loss',
        title='Training loss')
ax1.grid()
fig1.savefig(f'{path_to_output}/training_loss_test{test_number}.png')


fig2, ax2 = plt.subplots()
ax2.set_ylim([0, 1.1])
ax2.plot(epoch_iterations, np.array(training_loss_per_epoch), label='Training loss', marker='o')
ax2.plot(epoch_iterations, np.array(validation_loss_per_epoch), label='Validation loss', linestyle='dashed',
        marker='v')
ax2.set(xlabel='Number of epochs', ylabel=r'Normalized $L^2$ loss',
        title='Training and validation loss per epoch')
ax2.legend()
ax2.grid()
fig2.savefig(f'{path_to_output}/validation_loss_test{test_number}.png')

fig3, ax3 = plt.subplots()
ax1.set_ylim([0, 1.1])
ax3.set_yscale('log')
ax3.plot(epoch_iterations, np.array(training_loss_per_epoch), label='Training loss', marker='o')
ax3.plot(epoch_iterations, np.array(validation_loss_per_epoch), label='Validation loss', linestyle='dashed',
        marker='v')
ax3.set(xlabel='Number of epochs', ylabel=r'Normalized $L^2$ loss',
        title='Log of training and validation loss per epoch')
ax3.legend()
ax3.grid()
fig3.tight_layout()
fig3.savefig(f'{path_to_output}/log_loss{test_number}.png')
#plt.show()
plt.close()