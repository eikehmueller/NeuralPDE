import numpy as np
import torch
from firedrake import *
import sys
sys.path.append("..")
from neural_pde.icosahedral_dual_mesh import IcosahedralDualMesh
from neural_pde.katies_decoder import KatiesDecoder
import sys
from torch.profiler import profile, ProfilerActivity, record_function

def project_to_dg0(mesh, u_dual):
    """Project data from dual mesh to DG0 space on original mesh
    
    :arg mesh: orignal mesh
    :arg u_dual: vector of length n, where n is the number of vertices of the dual mesh,
                 whcih equals the number of cells of the original mesh

    returns a DG0 function on the original mesh
    """
    fs = FunctionSpace(mesh,"DG",0)
    u = Function(fs)
    plex = mesh.topology_dm
    section = fs.dm.getDefaultSection()
    pCellStart, pCellEnd = plex.getHeightStratum(0)
    n_cell = 20
    for cell in range(pCellStart,pCellEnd):
        offset = section.getOffset(cell)
        #u.dat.data[offset] = u_dual[cell-pCellStart]

        u.dat.data[:n_cell//2] = 1
    return u

mesh = UnitIcosahedralSphereMesh(refinement_level=3)
dualmesh = IcosahedralDualMesh(nref=0)
org_dualmesh = UnitIcosahedralSphereMesh(refinement_level=0)
V = FunctionSpace(mesh, "CG", 1)

dual_mesh_vertices = dualmesh.vertices
#z_prime = torch.rand(1, len(dual_mesh_vertices), 1)

z_prime = torch.tensor([[[0.], [0.], [1.], [1.],[0.], [0.],[0.], [0.],[0.], [0.],[0.], [1.],[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]])

d_lat = 1
n_ancil = 3
batch_size = 1
nu = 1

decoder_test = KatiesDecoder(V, dualmesh, 1, 1)
z = decoder_test.forward(z_prime, 1)
f_z = Function(V)

plex = mesh.topology_dm
section = V.dm.getDefaultSection()
pCellStart, pCellEnd = plex.getHeightStratum(0)
z_new = z[0].squeeze().detach().numpy()
print(f'Length of z_new is {len(z_new)}')
print(f'Length of f_z is {len(f_z.dat.data)}')
print(f'pCellEnd is {pCellEnd}')

for cell in range(pCellStart,pCellEnd):
    offset = section.getOffset(cell)
    f_z.dat.data[offset] = z_new[cell-pCellStart]

f_z.dat.data[:] = z[0].squeeze().detach().numpy()

u_dual = z_prime[0].squeeze()  # the original function on the dual mesh
u = project_to_dg0(org_dualmesh, u_dual)

file1 = VTKFile("output1.pvd")
file1.write(u)

file2 = VTKFile("output2.pvd")
file2.write(f_z)


'''
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        decoder_test.forward(z_prime, 1)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
with profile(
    activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True
) as prof:
    decoder_test.forward(z_prime, 1)

print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
prof.export_chrome_trace("trace.json")

'''

