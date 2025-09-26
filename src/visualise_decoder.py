import torch
from firedrake import *
from neural_pde.icosahedral_dual_mesh import IcosahedralDualMesh
from neural_pde.decoder import Decoder


nref = 5
nref_dual = 1
batchsize = 1

mesh = UnitIcosahedralSphereMesh(nref)
dual_mesh = IcosahedralDualMesh(nref_dual)

V = FunctionSpace(mesh, "CG", 1)
V_DG = FunctionSpace(mesh, "DG", 0)

linear_layer = torch.nn.Linear(in_features=2, out_features=1, bias=False)
linear_layer.weight = torch.nn.Parameter(torch.tensor([[1.0, 0.0]]))
decoder_model = torch.nn.Sequential(linear_layer)

decoder = Decoder(V, dual_mesh, decoder_model, nu=1)

rng = torch.Generator()
rng.manual_seed(1251527)
z_prime = torch.rand(size=(batchsize, len(dual_mesh.vertices), 1), generator=rng)
x_ancil = torch.zeros(size=(batchsize, 1, V.dof_count))

y = decoder(z_prime, x_ancil)

u = Function(V, name="decoded")
u.dat.data[:] = y.detach().numpy()

u_dg = dual_mesh.project_to_dg0(z_prime.detach().numpy().flatten())

file = VTKFile("output_decoded.pvd")
file.write(u)

file = VTKFile("output_dual.pvd")
file.write(u_dg)
