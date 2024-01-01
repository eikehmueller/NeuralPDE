import torch
from torch.utils.data import DataLoader

from firedrake import (
    UnitIcosahedralSphereMesh,
    Function,
    FunctionSpace,
    SpatialCoordinate,
)

from neural_pde.spherical_patch_covering import SphericalPatchCovering
from neural_pde.patch_encoder import PatchEncoder
from neural_pde.patch_decoder import PatchDecoder
from neural_pde.data_generator import AdvectionDataset

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
n_output = 2

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

# patch encoder
patch_encoder = PatchEncoder(
    V,
    spherical_patch_covering,
    dynamic_encoder_model,
    ancillary_encoder_model,
    n_dynamic,
)

# patch decoder
patch_decoder = PatchDecoder(
    V,
    spherical_patch_covering,
    decoder_model,
)

degree = 4
nsamples = 16
batchsize = 8

dataset = AdvectionDataset(V, nsamples, degree)

dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
for k, batched_sample in enumerate(iter(dataloader)):
    X, y = batched_sample
    z = patch_encoder(X)
    output = patch_decoder(z)
    print(k, "X: ", X.shape, "y: ", y.shape, "z: ", z.shape, "output: ", output.shape)
