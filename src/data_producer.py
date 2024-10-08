## This function produces the data
## This saves it in the form 'data_batchsize_nref_phi'
## Please change the seed every time so that we get different data

from neural_pde.data_classes import AdvectionDataset
from firedrake import FunctionSpace, UnitIcosahedralSphereMesh
import torch
import os

if not os.path.exists("data"):
    os.makedirs("data")

phi = 0.7854             # rotation angle of the data
degree = 4               # degree of the polynomials on the dataset
batchsize = 1024         # batchsize
n_ref = 2                # number of refinements in the icosahedral mesh
valid_batchsize = 64     # batchsize for validation group

mesh = UnitIcosahedralSphereMesh(n_ref) # create the mesh
V = FunctionSpace(mesh, "CG", 1) # define the function space

train_ds = AdvectionDataset(V, batchsize, phi, degree, seed=12344)
train_ds.generate()
train_ds.save(f"data/data_{batchsize}_{n_ref}_{phi}.npy")

valid_ds = AdvectionDataset(V, 32, phi, degree, seed=1237)  
valid_ds.generate()
valid_ds.save(f"data/data_{valid_batchsize}_{n_ref}_{phi}.npy")
