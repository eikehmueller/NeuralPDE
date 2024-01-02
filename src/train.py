import torch
from torch.utils.data import DataLoader

from firedrake import (
    UnitIcosahedralSphereMesh,
    FunctionSpace,
)

from neural_pde.spherical_patch_covering import SphericalPatchCovering
from neural_pde.patch_encoder import PatchEncoder
from neural_pde.patch_decoder import PatchDecoder
from neural_pde.data_generator import AdvectionDataset
from neural_pde.neural_solver import NeuralSolver

# construct spherical patch covering
spherical_patch_covering = SphericalPatchCovering(0, 4)

print(f"number of patches               = {spherical_patch_covering.n_patches}")
print(f"patchsize                       = {spherical_patch_covering.patch_size}")
print(f"number of points in all patches = {spherical_patch_covering.n_points}")

mesh = UnitIcosahedralSphereMesh(1)
V = FunctionSpace(mesh, "CG", 1)

# number of dynamic fields: scalar tracer
n_dynamic = 1
# number of ancillary fields: x-, y- and z-coordinates
n_ancillary = 3
# dimension of latent space
latent_dynamic_dim = 17
# dimension of ancillary space
latent_ancillary_dim = 6
# number of output fields: scalar tracer
n_output = 1

# encoder models
# dynamic encoder model: map all fields to the latent space
# input:  (n_dynamic+n_ancillary, patch_size)
# output: (latent_dynamic_dim)
dynamic_encoder_model = torch.nn.Sequential(
    torch.nn.Flatten(start_dim=-2, end_dim=-1),
    torch.nn.Linear(
        in_features=(n_dynamic + n_ancillary) * spherical_patch_covering.patch_size,
        out_features=latent_dynamic_dim,
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
# input:  (d_latent,4)
# output: (d_latent^dynamic)
interaction_model = torch.nn.Sequential(
    torch.nn.Flatten(start_dim=-2, end_dim=-1),
    torch.nn.Linear(
        in_features=4 * (latent_dynamic_dim + latent_ancillary_dim),
        out_features=latent_dynamic_dim,
    ),
).double()

# dataset
degree = 4
nsamples = 16
batchsize = 8

dataset = AdvectionDataset(V, nsamples, degree)

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

dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

loss_fn = torch.nn.MSELoss()

nepoch = 2

loss_history = []

# main training loop
for epoch in range(nepoch):
    for i, sample_batched in enumerate(iter(dataset)):
        X, y_target = sample_batched
        optimizer.zero_grad()
        y = model(X)
        loss = loss_fn(y, y_target)
        loss.backward()
        optimizer.step()
    loss_history.append(loss.item())
    print(f"{epoch:6d}", loss.item())
