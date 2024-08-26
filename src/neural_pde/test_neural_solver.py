# This module is for testing the Neural solver module. The tests here are 

# 1: checking that the neighbouring cells are neighbouring cells
# 2: checking that both neural solvers give the same shape
# 3: verifying that the averaging function works


from spherical_patch_covering import SphericalPatchCovering
from neural_solver import NeuralSolver, Katies_NeuralSolver
from data_generator import AdvectionDataset
import pytest
import torch
from firedrake import (
    UnitIcosahedralSphereMesh,
    FunctionSpace,
)


@pytest.fixture
def mesh_data():
    return

spherical_patch_covering = SphericalPatchCovering(0, 4)
num_ref=1
mesh = UnitIcosahedralSphereMesh(num_ref) # create the mesh
V = FunctionSpace(mesh, "CG", 1) # define the function space
dataset = AdvectionDataset(V, 7, 1, degree=4)
#sample_batched =  dataset[0] # sample dof vector
#print(sample_batched.shape)
sample = dataset[0][0] # has u, x, y, z
print(sample.shape) # this is the dof map for one sample
#self.spherical_patch_covering.neighbour_list 

latent_dynamic_dim = 7 
latent_ancillary_dim = 3 

interaction_model = torch.nn.Sequential(
    torch.nn.Flatten(start_dim=-2, end_dim=-1),
    torch.nn.Linear(
        in_features=4 * (latent_dynamic_dim + latent_ancillary_dim), # do we use a linear model here?? Or do we need a nonlinear part
        out_features=latent_dynamic_dim,
    ),
).double()

def test_same_length_batches(NeuralSolver, Katies_NeuralSolver, sample):

    model1 = NeuralSolver(
        spherical_patch_covering,
        interaction_model,
        nsteps=1,
        stepsize=1.0,
    )

    model2 = Katies_NeuralSolver(
        spherical_patch_covering,
        interaction_model,
        nsteps=1,
        stepsize=1.0,
    )

    y1 = model1(sample)
    y2 = model2(sample)

    assert y1.shape == y2.shape 

def test_is_neighbour():
    spherical_patch_covering = SphericalPatchCovering(0, 4)
    mesh = UnitIcosahedralSphereMesh(num_ref) # create the mesh
    V = FunctionSpace(mesh, "CG", 1) # define the function space

    neighbour_list = spherical_patch_covering.neighbour_list 
    plex = mesh.topology_dm

    for i in range(pCellStart, pCellEnd):
        for j in range(3):
            assert()
    #vertex_coords = np.asarray(plex.getCoordinates()).reshape([-1, dim])
    pCellStart, pCellEnd = plex.getHeightStratum(0)
    pEdgeStart, pEdgeEnd = plex.getHeightStratum(1)
    pVertexStart, _ = plex.getHeightStratum(2)

    # work out the cell centres
    self._cell_centres = np.zeros((pCellEnd - pCellStart, 3))
    for cell in range(pCellStart, pCellEnd):
        vertices = {
            vertex
            for edge in plex.getCone(cell).tolist()
            for vertex in plex.getCone(edge).tolist()
        }
        p = np.asarray(
            [
                vertex_coords[vertex - pVertexStart, :].tolist()
                for vertex in vertices
            ]
        )
        self._cell_centres[cell, :] = np.average(p[:, :], axis=0)
    # work out the connectivity information
    neighbour_map = defaultdict(set)
    for edge in range(pEdgeStart, pEdgeEnd):
        cells = [cell - pCellStart for cell in plex.getSupport(edge).tolist()]
        for cell in cells:
            neighbour_map[cell].update([other for other in cells if other != cell])
    self._neighbours = [list(value) for key, value in sorted(neighbour_map.items())]
    return 


test_same_length_batches(NeuralSolver, Katies_NeuralSolver, sample)

def test_same_length_single():
    return

def test_simple_function():
    return 