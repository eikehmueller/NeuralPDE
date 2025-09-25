import numpy as np
import torch
from firedrake import *
import sys
sys.path.append("..")
from neural_pde.icosahedral_dual_mesh import IcosahedralDualMesh
from neural_pde.katies_decoder import KatiesDecoder
import sys

mesh = UnitIcosahedralSphereMesh(refinement_level=0)
dualmesh = IcosahedralDualMesh(nref=0)
V = FunctionSpace(mesh, "CG", 1)

dual_mesh_vertices = dualmesh.vertices
print(len(dual_mesh_vertices))
z_prime = torch.rand(1, len(dual_mesh_vertices), 1)

d_lat = 1
n_ancil = 3
batch_size = 1
nu = 1

decoder_test = KatiesDecoder(V, dualmesh, 1, 1)

z = decoder_test.forward(z_prime, 1)
print(z)

'''
V = FunctionSpace(mesh, "CG", 1), 1
for j, ((X, t), y_target) in enumerate(iter(dataset)):

    f_input = Function(V, name="input")
    f_input.dat.data[:] = X.detach().numpy()[0, :]

    f_target = Function(V, name="target")
    f_target.dat.data[:] = y_target.detach().numpy()[0, :]

    file = VTKFile(os.path.join(args.output, f"output_{j:04d}.pvd"))
    file.write(f_input, f_target)

'''
