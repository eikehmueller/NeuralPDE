"""Decode based on the latent state and ancillary state"""

import numpy as np
import torch

from firedrake import *

__all__ = ["Decoder"]


class Decoder(torch.nn.Module):
    """Class representing the decoder network

    the model is fed two tensors Y' and X:

       Z':        latent state, tensor of shape (batch_size, n_dual_vertex, d_lat)
       X^{ancil}: ancillary state, tensor of shape (batch_size, n_vertex, d_ancillary)

    The tensor Z' is gathered to a tensor tilde{Z} of shape (batch_size, n_vertex, nu*d_lat), which
    for each vertex in the mesh contains the latent state of the nu dual vertices that are
    closest to  the vertex.

    The tensors tilde{Z} and X^{ancil} are concatenated to a tensor tilde{Y} of shape
    (batch_size, nu * d_lat + d_ancillary, n_vertex). This is passed to the decoder model
    which returns a tensor Y of shape (batch_size,n_out,n_vertex).
    """

    def __init__(self, V, dual_mesh, decoder_model, nu=3):
        """Initialise new instance

        :arg V: function space
        :arg dual_mesh: dual mesh which defines the latent space
        :arg decoder_model: decoder model
        :arg nu: number of dual vertices to gather information from
        """
        super().__init__()
        mesh = V.mesh()
        self._decoder_model = decoder_model
        self.nu = nu
        plex = mesh.topology_dm
        dim = plex.getCoordinateDim()
        vertex_coords = np.asarray(plex.getCoordinates()).reshape([-1, dim])
        n_vertex = vertex_coords.shape[0]
        pVertexStart, pVertexEnd = plex.getHeightStratum(2)
        section = V.dm.getDefaultSection()
        # Construct index array of shape (n_vertex, nu) such that
        # index[j,k] = index of the k-th closest dual vertex to vertex j
        index = np.zeros(shape=(n_vertex, self.nu), dtype=np.int64)
        for vertex in range(pVertexStart, pVertexEnd):
            offset = section.getOffset(vertex)
            p = vertex_coords[vertex - pVertexStart, :]
            dist = np.linalg.norm(dual_mesh._vertices - p, axis=1)
            for k in range(self.nu):
                idx = np.argmin(dist)
                index[offset, k] = idx
                dist[idx] = np.inf
        self.register_buffer("index", torch.tensor(index).unsqueeze(-1))

    def forward(self, z_prime, x_ancil):
        """Forward pass of the decoder

        :arg z_prime: latent state, tensor of shape (batch_size, n_dual_vertex, d_lat)
        :arg x_ancil: ancillary state, tensor of shape (batch_size, n_ancil, n_vertex)
        """
        # expand index to shape (batch_size, n_vertex, nu, d_lat)
        index = self.index.expand(z_prime.shape[:-2] + (-1, -1, z_prime.shape[-1]))
        dim = z_prime.dim()
        # gather z_prime to shape (batch_size, n_vertex, nu,d_lat) and flatten final two
        # dimensions to shape (batch_size, n_vertex, nu*d_lat)
        z_tilde = torch.gather(
            z_prime.unsqueeze(-2).repeat((dim - 1) * (1,) + (self.nu, 1)),
            dim - 2,
            index,
        ).flatten(start_dim=-2, end_dim=-1)
        # Convert x_ancil to shape (batch_size, n_vertex, n_ancil) by swapping the last two
        # dimensions
        x_ancil_tilde = x_ancil.transpose(-2, -1)
        # Concat z_tilde and x_ancil_tilde to shape (batch_size, nu*d_lat + n_ancil, n_vertex)
        y_tilde = torch.cat((z_tilde, x_ancil_tilde), dim=-1)
        # Pass through decoder model and transpose to shape (batch_size, n_out, n_vertex)
        return self._decoder_model(y_tilde).transpose(-2, -1)
    
    def test_forward(self, z_prime):
        """Forward pass of the decoder

        :arg z_prime: latent state, tensor of shape (batch_size, n_dual_vertex, d_lat)
        :arg x_ancil: ancillary state, tensor of shape (batch_size, n_ancil, n_vertex)
        """
        # expand index to shape (batch_size, n_vertex, nu, d_lat)
        index = self.index.expand(z_prime.shape[:-2] + (-1, -1, z_prime.shape[-1]))
        dim = z_prime.dim()
        # gather z_prime to shape (batch_size, n_vertex, nu,d_lat) and flatten final two
        # dimensions to shape (batch_size, n_vertex, nu*d_lat)
        z_tilde = torch.gather(
            z_prime.unsqueeze(-2).repeat((dim - 1) * (1,) + (self.nu, 1)),
            dim - 2,
            index,
        ).flatten(start_dim=-2, end_dim=-1)
        # Convert x_ancil to shape (batch_size, n_vertex, n_ancil) by swapping the last two
        # dimensions
        
        return z_tilde
