"""Decode based on the latent state and ancillary state"""

import numpy as np
import torch
from neural_pde.icosahedral_dual_mesh import IcosahedralDualMesh

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
        """
        super().__init__() # initialise the methods from the parent class (torch.nn.Module)
                          # This means that the Decoder is a neural network Module

        mesh = V.mesh() # mesh underlying the function space V
        #self._decoder_model = decoder_model # couple the decoder model to the class Instance
        self.nu = nu # couple number of neighbours to class Instance
        plex = mesh.topology_dm # PETSc.DMPlex representing the mesh topology
        dim = plex.getCoordinateDim() # dimensions of the embedding space (3 dimension on a sphere)
        coords = plex.getCoordinates() # get global vector with coordinates associated with the plex
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


    def testing(self):
        return
    
    def forward(self, z_prime, x_ancil):
        """
        Forward pass of the decoder model

        :arg z_prime: latent state, tensor of shape (batch_size, n_dual_vertex, d_lat)
        :arg x_ancil: ancillary state, tensor of shape (batch_size, n_ancil, n_vertex)
        """

        z_bar = torch.zeros_like(self.index, dtype=torch.float64)

        z_tilde = z_bar.repeat(z_prime.shape[0], 1, 1, z_prime.shape[-1])


        print(z_tilde.shape)
        # i is the firedrake node
        # j is how close the node is
        # k is the barch size 
        for i in range(z_tilde.shape[1]):
            for j in range(self.nu):
                for k in range(z_tilde.shape[0]):
                    index_value = self.index[i][j] # 0 corresponds to the nearest neighbour
                    #print(f'Index value is {index_value}')
                    #print(f'We want to insert {z_prime[index_value].squeeze()}')
                    #print(f'We will insert it at {z_tilde[i][0]}')
                    z_tilde[k][i][j] = z_prime[k][index_value].squeeze().detach().clone()
                    #print(z_tilde[0][0])
                    #z_tilde[i][0] = torch.ones(5)

        z_tilde = z_tilde.flatten(start_dim=-2, end_dim=-1)
        #x_ancil_tilde = x_ancil.transpose(-2, -1)
        #y_tilde = torch.cat((z_tilde, x_ancil_tilde), dim=-1)

        return z_tilde #self._decoder_model(y_tilde).transpose(-2, -1)

'''

d_lat = 5
n_ancil = 3
batch_size = 8 

icomesh = UnitIcosahedralSphereMesh(refinement_level=0)
dualmesh = IcosahedralDualMesh(nref=0)
V = FunctionSpace(icomesh, "CG", 1)  # define the function space

n_dual_vertex = dualmesh.dof_count
#print(f'No dual vertices is {n_dual_vertex}')

n_vertex = V.dim()
#print(f'No vertices of firedrake mesh is {n_vertex}')

z_prime = torch.rand(batch_size, n_dual_vertex, d_lat)
#print(f'Shape of z_prime is {z_prime.shape}')
x_ancil = torch.rand(batch_size, n_ancil, n_vertex)
#print(f'Shape of x_ancil is {x_ancil.shape}')
'''
#testing_decoder = KatiesDecoder(V, dualmesh, 1)
#testing_decoder.forward(z_prime, x_ancil)