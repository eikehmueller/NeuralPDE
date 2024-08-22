from spherical_patch_covering import SphericalPatchCovering
import pytest

from firedrake import (
    UnitIcosahedralSphereMesh,
    FunctionSpace,
)


#@pytest.fixture
def mesh_data():
    return

#spherical_patch_covering = SphericalPatchCovering(0, 4)
#mesh = UnitIcosahedralSphereMesh(2) # create the mesh
#V = FunctionSpace(mesh, "CG", 1) # define the function space

#self.spherical_patch_covering.neighbour_list 

def test_is_neighbour():
    spherical_patch_covering = SphericalPatchCovering(0, 4)
    mesh = UnitIcosahedralSphereMesh(2) # create the mesh
    V = FunctionSpace(mesh, "CG", 1) # define the function space

    neighbour_list = spherical_patch_covering.neighbour_list 
    

    for i in range(mesh.num_cells()):
        facets = mesh.cell_to_facets[i]
        print(facets)
    '''
    dim = plex.getCoordinateDim()
    vertex_coords = np.asarray(plex.getCoordinates()).reshape([-1, dim])
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
    '''
    return 




test_is_neighbour()




def test_simple_function():
    return 