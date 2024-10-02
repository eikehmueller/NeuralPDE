## This function produces the data
## This saves it in the form 'data_batchsize_nref_phi'
## Please change the seed every time so that we get different data

from neural_pde.data_classes import AdvectionDataset
from firedrake import FunctionSpace, UnitIcosahedralSphereMesh
import torch
import os

if not os.path.exists("data"):
    os.makedirs("data")

phi = 1                  # rotation angle of the data
degree = 4               # degree of the polynomials on the dataset

mesh = UnitIcosahedralSphereMesh(2) # create the mesh
V = FunctionSpace(mesh, "CG", 1) # define the function space

train_ds = AdvectionDataset(V, 512, phi, degree, seed=12345)
train_ds.generate()
train_ds.save("data/data_512_2_1.npy")

valid_ds = AdvectionDataset(V, 32, phi, degree, seed=123)  
valid_ds.generate()
valid_ds.save("data/data_32_2_1.npy")
