from collections import defaultdict
import numpy as np
from firedrake import *


class IcosahedralDualMesh:
    """Class that defines the dual of an icosahedral mesh on the unit sphere.

    :arg nref: number of refinements of the icosahedral mesh
    """

    def __init__(self, nref):
        self.nref = nref
        #mesh = UnitIcosahedralSphereMesh(nref)
        #plex = mesh.topology_dm
        self._mesh = UnitIcosahedralSphereMesh(nref)
        plex = self._mesh.topology_dm
        dim = plex.getCoordinateDim()
        vertex_coords = np.asarray(plex.getCoordinates()).reshape([-1, dim])
        pCellStart, pCellEnd = plex.getHeightStratum(0)
        pEdgeStart, pEdgeEnd = plex.getHeightStratum(1)
        pVertexStart, _ = plex.getHeightStratum(2)

        # work out the cell centres
        self._vertices = np.zeros((pCellEnd - pCellStart, 3))
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
            self._vertices[cell, :] = np.average(p[:, :], axis=0)
        # work out the connectivity information
        neighbour_map = defaultdict(set)
        for edge in range(pEdgeStart, pEdgeEnd):
            cells = [cell - pCellStart for cell in plex.getSupport(edge).tolist()]
            for cell in cells:
                neighbour_map[cell].update([other for other in cells if other != cell])
        self._neighbours = [list(value) for key, value in sorted(neighbour_map.items())]

    @property
    def vertices(self):
        """return an array of shape (n_cells,3) which contains the vertices of the dual grid"""
        return np.array(self._vertices)
    
    @property
    def dof_count(self):
        """Return the number of dofs on the dual mesh"""
        return len(self._vertices[:,0])

    @property
    def neighbour_list(self):
        """return neighbour list"""
        return list(self._neighbours)
    
    def project_to_dg0(self, u_dual):

        """Project data from dual mesh to DG0 space on original mesh
        :arg u_dual: vector of length n, where n is the number of vertices of the dual mesh,
                     which equals the number of cells of the original mesh

        returns a DG0 function on the original mesh


        """
        fs = FunctionSpace(self._mesh, "DG", 0)
        u = Function(fs, name="target")
        plex = self._mesh.topology_dm
        section = fs.dm.getDefaultSection()
        pCellStart, pCellEnd = plex.getHeightStratum(0)
        for cell in range(pCellStart, pCellEnd):
            offset = section.getOffset(cell)
            u.dat.data[offset] = u_dual[cell - pCellStart]
        return u
