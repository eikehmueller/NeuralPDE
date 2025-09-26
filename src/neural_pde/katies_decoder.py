"""Decode based on the latent state and ancillary state"""

import numpy as np
import torch
from firedrake.__future__ import interpolate

from firedrake import *

__all__ = ["Decoder"]


class KatiesDecoder(torch.nn.Module):

    def __init__(self, V, dual_mesh, decoder_model, nu=3):
        """
        Initialise a new instance of the decoder class

        :arg V: the Function space we are interpolating onto
        :arg dual_mesh: instance of IcosahedralDualMesh class
        :decoder_model: sequential pytorch model
        :nu: the number of neighbours to draw from

        This is now modified to take any funciton space!
        """
        super().__init__() # initialise the methods from the parent class (torch.nn.Module)
                          # This means that the Decoder is a neural network Module

        mesh = V.mesh() # mesh underlying the function space V
        #print(f'Dimension of V is {V.dim()}')
        self._decoder_model = decoder_model # couple the decoder model to the class Instance
        self.nu = nu # couple number of neighbours to class Instance
        plex = mesh.topology_dm # PETSc.DMPlex representing the mesh topology
        dim = plex.getCoordinateDim() # dimensions of the embedding space (3 dimension on a sphere)
        coords = plex.getCoordinates() # get global vector with coordinates associated with the plex
        #print(f'Dof count is {V.dof_count}')
        #print(f'Length of coordinates are {len(np.asarray(coords))}')
        #W = VectorFunctionSpace(mesh, V.ufl_element())
        #interpolated = interpolate(mesh.coordinates, W)
        #DG_coords = assemble(interpolate(mesh.coordinates, W))
        #print(f'DG coords are {DG_coords}')
        # This is one long array, so we need to reshape it into a n_dofs x dim array
        vertex_coords = np.asarray(coords).reshape([-1, dim])
        n_vertex = vertex_coords.shape[0] # the number of vertices in the icosahedral mesh
        index = np.zeros(shape=(n_vertex, self.nu), dtype=np.int64) # empty array
        for j in range(n_vertex):
            p = vertex_coords[j]
            dist = np.linalg.norm(dual_mesh.vertices - p, axis=1) 
            for k in range(self.nu):
                idx = np.argmin(dist)
                index[j, k] = idx
                dist[idx] = np.inf

        self.register_buffer("index", torch.tensor(index).unsqueeze(-1))
    
    def forward(self, z_prime, x_ancil):
        """
        Forward pass of the decoder model

        :arg z_prime: latent state, tensor of shape (batch_size, n_dual_vertex, d_lat)
        :arg x_ancil: ancillary state, tensor of shape (batch_size, n_ancil, n_vertex)
        """

        # create an zero tensor with shape (n_vertex, nu)
        z_bar = torch.zeros_like(self.index, dtype=torch.float32)

        # expand to be zero tensor of shape (batch_size, n_vertex, nu, d_lat)
        z_tilde = z_bar.repeat(z_prime.shape[0], 1, 1, z_prime.shape[-1])



        # i is the firedrake node
        # j is how close the node is
        # k is the branch size 
        for k in range(z_tilde.shape[0]):
            for i in range(z_tilde.shape[1]):
                for j in range(self.nu):
                    # jth nearest neighbour to ith node
                    index_value = self.index[i][j] # 
                    # fill z_tilde batch k, node i, jth nearest neighbour
                    # with the d_latent variable data
                    z_tilde[k][i][j] = z_prime[k][index_value].squeeze().detach().clone()
        
        #print(f'Shape of z_tilde is {z_tilde.size()}')
        # combine nu and dlat 
        z_tilde = z_tilde.flatten(start_dim=-2, end_dim=-1)
        x_ancil_tilde = x_ancil.transpose(-2, -1)
        y_tilde = torch.cat((z_tilde, x_ancil_tilde), dim=-1)
        print(type(z_tilde))
        print(type(x_ancil_tilde))

        return self._decoder_model(y_tilde).transpose(-2, -1)