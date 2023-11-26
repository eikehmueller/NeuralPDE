from collections import defaultdict
import numpy as np
import plotly.graph_objects as go
import scipy as sp
from firedrake import *


class SphericalPatch:
    """A single spherical patch

    a patch is defined by a collection of points around a central point
    on the sphere. The points are arranged in circles around the central
    point and their distance is adjusted to be approximately uniform and
    of around dtheta = r_theta/n_theta.
    """

    def __init__(self, n_normal, r_theta, n_theta):
        """Initialise new instance

        :arg n_normal: vector defining the centre of the patch, will
                       be normalised to have length 1
        :arg r_theta: radius of the patch
        :arg n_theta: number of points used to discretise in the theta-direction
        """
        # construct unit normal vector that defines the centre of the patch
        self._n_normal = np.array(n_normal)
        assert np.linalg.norm(self._n_normal) > 1.0e-12
        self._n_normal /= np.linalg.norm(self._n_normal)
        # construct points in 2d plane
        points_2d = [[0, 0]]
        for k in range(1, n_theta + 1):
            theta = k / n_theta * r_theta
            n_phi = int(2 * np.pi * np.sin(theta) // (r_theta / n_theta))
            Phi = np.linspace(0, 2 * np.pi * (1 - 1 / n_phi), n_phi)
            points_2d += [[theta, phi] for phi in Phi]
        coords_2d = np.asarray(points_2d)
        Theta = coords_2d[:, 0]
        Phi = coords_2d[:, 1]
        # construct spherical patch around the north pole by projecting the
        # 2d points
        p = (
            np.array(
                [
                    np.sin(Theta) * np.cos(Phi),
                    np.sin(Theta) * np.sin(Phi),
                    np.cos(Theta),
                ]
            )
            .reshape([3, -1])
            .transpose()
        )
        # rotate to the position defined by the normal vector
        north_pole = np.array([0, 0, 1])
        if np.linalg.norm(self._n_normal - north_pole) < 1.0e-14:
            self._points = p
        else:
            rot, _ = sp.spatial.transform.Rotation.align_vectors(
                [self._n_normal],
                [north_pole],
            )
            self._points = rot.apply(p)

    @property
    def points(self):
        """Return the array of points in the patch

        This returns an array of shape (n_points,3)
        """
        return self._points


class SphericalPatchCovering:
    """Class that defines a coverging of the unit sphere with spherical patches

    The centres of the patches are arranged to form the dual of a refined
    icosahedral mesh, which implies that each patch has exactly three neighbours.

    :arg nref: number of refinements of the icosahedral mesh
    :arg patch_radius: radius of the patches
    :arg patch_n: number of radial points in each patch
    """

    def __init__(self, nref, patch_radius, patch_n):
        self.patch_radius = patch_radius
        self.patch_n = patch_n

        mesh = UnitIcosahedralSphereMesh(nref)
        plex = mesh.topology_dm
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
        self._patches = [
            SphericalPatch(self._cell_centres[j, :], patch_radius, patch_n)
            for j in range(self._cell_centres.shape[0])
        ]

    def visualise(self):
        """Visualise spherical patch covering

        plots the mesh (vertices and edges) and the patch coverings
        """
        fig = go.Figure()
        # extract vertices
        vertices = self._cell_centres[:, :]
        # vertices of mesh
        fig.add_trace(
            go.Scatter3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                mode="markers",
                marker_color="blue",
            ),
        )
        # edges of mesh
        for j, nbs in enumerate(self._neighbours):
            for k in nbs:
                fig.add_trace(
                    go.Scatter3d(
                        x=[vertices[j, 0], vertices[k, 0]],
                        y=[vertices[j, 1], vertices[k, 1]],
                        z=[vertices[j, 2], vertices[k, 2]],
                        mode="lines",
                        line_color="blue",
                    ),
                )
        # patches
        for patch in self._patches:
            points = patch.points
            fig.add_trace(
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    marker_color="red",
                    marker_size=2,
                ),
            )
        fig.show()


sperhical_patch_covering = SphericalPatchCovering(1, 0.2, 4)
sperhical_patch_covering.visualise()
