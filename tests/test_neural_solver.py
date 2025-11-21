"""This module is for testing the Neural solver module. The tests here are

1: check that the neighbouring cells have a shared edge
2: check that input and output have shape (n_patches, d_lat)
3: check that input and output have shape (1, n_patches, d_lat)
4: check that input and output have shape (batchsize, n_patches, d_lat)
5: check that an averaging function returns the expected result
"""

from neural_pde.spherical_patch_covering import SphericalPatchCovering
from neural_pde.neural_solver import ForwardEulerNeuralSolver
import torch
from firedrake import UnitIcosahedralSphereMesh


###### setup #######

spherical_patch_covering = SphericalPatchCovering(0, 4)
n_patches = spherical_patch_covering.n_patches
mesh = UnitIcosahedralSphereMesh(0)  # create the mesh

# tensor shapes
batchsize = 16  # number of samples to use in each batch
latent_dynamic_dim = 7
latent_ancillary_dim = 3
d_lat = latent_dynamic_dim + latent_ancillary_dim
d_dyn = latent_dynamic_dim

sample_2d = torch.rand(n_patches, d_lat)  # A sample with shape (n_patches, d_lat)
sample_3d_flat = torch.rand(
    1, n_patches, d_lat
)  # A sample with shape (1, n_patches, d_lat)
sample_3d_full = torch.rand(
    batchsize, n_patches, d_lat
)  # A sample with shape (batchsize, n_patches, d_lat)
t_final_2d = torch.tensor(1.0)
t_final_3d = torch.ones(batchsize)

neighbour_list = spherical_patch_covering.neighbour_list

interaction_model = torch.nn.Sequential(
    torch.nn.Flatten(start_dim=-2, end_dim=-1),
    torch.nn.Linear(
        in_features=4
        * (
            latent_dynamic_dim + latent_ancillary_dim
        ),  # do we use a linear model here?? Or do we need a nonlinear part
        out_features=latent_dynamic_dim,
    ),
)


###### functions for the tests #######


def average_model(x):
    """For an array of shape (B, n_patch, n_func, 4), take the average
    across the final dimension of the tensor to return (Batch, n_patches, n_func_average)
    """
    summed = 0.25 * torch.sum(x, -1)

    return summed[:, :, :d_dyn]


def average_neighbours(x, neighbour_list):
    """Given an input of shape (Batch, n_patches, n_func),
    and a neighbourhood array, take the average over all connecting elements"""

    average = torch.zeros_like(x)

    for i in range(average.shape[0]):
        for j in range(average.shape[1]):
            for k in range(average.shape[2]):
                neighbours = neighbour_list[j]
                average[i, j, k] = 0.25 * (
                    x[i, j, k]
                    + x[i, neighbours[0], k]
                    + x[i][neighbours[1]][k]
                    + x[i][neighbours[2]][k]
                )
    return average


######## The tests #########


def test_is_neighbour():
    """Check that the neighbouring cells in neighbour_list have a shared edge"""

    plex = mesh.topology_dm
    num_errors = 0

    for cell in range(len(neighbour_list)):
        for neighbouring_cells in range(3):
            edge_of_cell = plex.getCone(cell).tolist()
            edge_of_neighbouring_cell = plex.getCone(
                neighbour_list[cell][neighbouring_cells]
            ).tolist()
            if any(edge in edge_of_cell for edge in edge_of_neighbouring_cell):
                pass
            else:
                num_errors += 1

    assert num_errors == 0


def test_sample_2d():
    """Check whether the input and output are the same shape"""
    model = ForwardEulerNeuralSolver(
        spherical_patch_covering,
        interaction_model,
        stepsize=1.0,
    )
    y = model(sample_2d, t_final_2d)

    assert y.shape == sample_2d.shape


def test_sample_3d_full():
    """Check whether the input and output are the same shape"""
    model = ForwardEulerNeuralSolver(
        spherical_patch_covering,
        interaction_model,
        stepsize=1.0,
    )
    y = model(sample_3d_full, t_final_3d)

    assert y.shape == sample_3d_full.shape


def test_average():
    """Check whether the model returns the expected result from an averaging function"""

    model = ForwardEulerNeuralSolver(
        spherical_patch_covering,
        average_model,
        stepsize=1.0,
    )

    y1 = model(sample_3d_full, t_final_3d)
    y2 = sample_3d_full

    y2[:, :, :d_dyn] += average_neighbours(sample_3d_full, neighbour_list)[:, :, :d_dyn]
    assert torch.allclose(y1, y2)
