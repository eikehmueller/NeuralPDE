import torch
from firedrake import *
from neural_pde.icosahedral_dual_mesh import IcosahedralDualMesh
from neural_pde.decoder import Decoder

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
    for cell in range(pCellStart,pCellEnd):
        offset = section.getOffset(cell)
        u.dat.data[offset] = u_dual[cell-pCellStart]
    return u

mesh = UnitIcosahedralSphereMesh(refinement_level=4) # the firedrake mesh
dualmesh = IcosahedralDualMesh(nref=0) # the dual mesh (much coarser)
V = FunctionSpace(mesh, "CG", 1) # FEM function space related to firedrake mesh

dual_mesh_vertices = dualmesh.vertices 

z_prime = torch.rand(1, len(dual_mesh_vertices), 1)
u_dual = z_prime[0].squeeze()  # the original function on the dual mesh
u = dualmesh.project_to_dg0(u_dual) # define a dg firedrake function that can be visualised

decoder_model = 1 # not used in function, so we set it equal to one
nu = 1 # only one closest variable

decoder_test = Decoder(V, dualmesh, decoder_model, nu) # initialise first class

z = decoder_test.test_forward(z_prime) # decode from z prime
z_new = z[0].squeeze().detach().numpy() # change to numpy array

f_z = Function(V, name="decoder_output") # write to firedrake function
f_z.dat.data[:] = z_new

# These functions can be visualised and inspected on paraview
file1 = VTKFile("Test_Decoder_Target.pvd")
file1.write(u)

file1 = VTKFile("Test_Decoder_Output.pvd")
file1.write(f_z)