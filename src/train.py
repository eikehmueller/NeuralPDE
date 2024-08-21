import torch
from torch.utils.data import DataLoader

from firedrake import (
    UnitIcosahedralSphereMesh,
    FunctionSpace,
    Function,
    VTKFile
)

print('Testing for wsl')

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("tensorboard_logs/solid_body_rotation_experiment_1")

from neural_pde.spherical_patch_covering import SphericalPatchCovering
from neural_pde.patch_encoder import PatchEncoder
from neural_pde.patch_decoder import PatchDecoder
from neural_pde.data_generator import AdvectionDataset
from neural_pde.neural_solver import NeuralSolver

# construct spherical patch covering with
# arg1: number of refinements of the icosahedral sphere
# arg2: number of radial points on each patch
spherical_patch_covering = SphericalPatchCovering(0, 4)

print(f"number of patches               = {spherical_patch_covering.n_patches}")
print(f"patchsize                       = {spherical_patch_covering.patch_size}")
print(f"number of points in all patches = {spherical_patch_covering.n_points}")

mesh = UnitIcosahedralSphereMesh(2) # create the mesh
V = FunctionSpace(mesh, "CG", 1) # define the function space

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
phi = 1 # rotation angle of the data
degree = 4 # degree of the polynomials on the dataset
n_train_samples = 100 # number of samples in the training dataset
n_valid_samples = 10
batchsize = 10 # number of samples to use in each batch

train_ds = AdvectionDataset(V, n_train_samples, phi, degree)
valid_ds = AdvectionDataset(V, n_valid_samples, phi, degree) 

train_dl = DataLoader(train_ds, batch_size=batchsize, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batchsize * 2)


# visualise the first object in the training dataset 
u_in = Function(V, name="input")
u_in.dat.data[:] = train_ds[0][0][0].numpy()
u_target = Function(V, name="target")
u_target.dat.data[:] = train_ds[0][1].numpy()
file = VTKFile("/home/katie795/internship/solid_body_rotation/output/training_example.pvd")
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
    NeuralSolver(spherical_patch_covering, interaction_model, nsteps=1, stepsize=1.0),
    PatchDecoder(V, spherical_patch_covering, decoder_model),
)

opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.1) 

loss_fn = torch.nn.MSELoss() # mean squared error loss function 
loss_history = []

nepoch = 5

# main training loop
for epoch in range(nepoch):
    for Xb, yb in train_dl:
        opt.zero_grad() # resets all of the gradients to zero, otherwise the gradients are accumulated
        y_pred = model(Xb)

        loss = loss_fn(y_pred, yb)
        loss.backward() 
        opt.step() # adjust the parameters by the gradient collected in the backwards pass
    writer.add_scalar("Loss/train", loss, epoch)
    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_fn(model(xb), yb) for xb, yb in valid_dl)
    loss_history.append(loss.item())
    print(f"{epoch:6d}", loss.item())

writer.flush()

# visualise the first object in the training dataset 
u_in = Function(V, name="input")
u_in.dat.data[:] = valid_ds[0][0][0].numpy()
u_target = Function(V, name="target")
u_target.dat.data[:] = valid_ds[0][1].numpy()
u_predicted_values = model(valid_ds[0][0].unsqueeze(0)).squeeze()
u_predicted = Function(V, name="predicted")
u_predicted.dat.data[:] = u_predicted_values.detach().numpy()
file = VTKFile("/home/katie795/internship/solid_body_rotation/output/validation_example.pvd")
file.write(u_in, u_target, u_predicted) 

patch_encoder = PatchEncoder(V,
        spherical_patch_covering,
        dynamic_encoder_model,
        ancillary_encoder_model,
        n_dynamic,)

#tensorboard session
dataiter = iter(train_dl)
X, y = next(dataiter)
writer.add_graph(model, X)
writer.close()

