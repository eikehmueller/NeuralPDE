"""Decode based on the latent state and ancillary state"""

import numpy as np
import torch
from neural_pde.icosahedral_dual_mesh import IcosahedralDualMesh

from firedrake import *

__all__ = ["Decoder"]


class Decoder(torch.nn.Module):

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
        #print(len(np.asarray(coords)))
        n_vertex = vertex_coords.shape[0] # the number of vertices in the icosahedral mesh
        #print(f'normal mesh vertices are {n_vertex}')

        # next what we need to do is find the closest dual point to each point in this mesh
        # index[j, k], gives the kth closest dual vertex to vertex j

        index = np.zeros(shape=(n_vertex, self.nu), dtype=np.int64) # empty array
        for j in range(n_vertex):
            p = vertex_coords[j]
            # distance between dual mesh vertices and that coordinate p 
            #print(f'dual mesh vertices are {dual_mesh.vertices}')
            #print(f'dual mesh dof count is {dual_mesh.dof_count}')
            #print(f'dual mesh neighbours list is {dual_mesh.neighbour_list}')
            #print(f'p is {p}')
            #print(f'Dual mesh distances are {dual_mesh.vertices - p}')
            dist = np.linalg.norm(dual_mesh.vertices - p, axis=1) 
            #print(f'distance is {dist}')
            # We now have an array with the distances between the vertex coords and the point p
            # but there are 6 vertices that are equidistant 
            for k in range(self.nu):
                idx = np.argmin(dist)
                #print(f'Idx is {idx}') # this is the index of the smallest one
                index[j, k] = idx
                dist[idx] = np.inf

        # now that we have the index array, we change it into a pytorch tensor
        # The .unsqueeze(-1) adds a [] around every element in the array
        # -1 means smallest array
        #print(torch.tensor(index).unsqueeze(-1))
        self.register_buffer("index", torch.tensor(index).unsqueeze(-1))
        #print(torch.tensor(index).unsqueeze(-1))

    def testing(self):
        return
    
    def forward(self, z_prime, x_ancil):
        """
        Forward pass of the decoder model

        :arg z_prime: latent state, tensor of shape (n_dual_vertex, d_lat)
        :arg x_ancil: ancillary state, tensor of shape (n_ancil, n_vertex)
        """
        # Expand index to shape (n_vertex, nu, d_lat)
        # What are we doing here?
        # basically we want to combine these two, and reqwrite an index?

        #print(f'z_prime shape -2 here is {z_prime.shape[:-2]}') # this does not work in two dimensions
        #print(f'z_prime shape -1 here is {z_prime.shape[-1]}')


        # first we expand index to the shape (n_vertex, nu, d_lat)
        print(self.index.size())
        # index has size (n_vertex, nu, 1)
        # not sure why we need the -2 one 

        # this is a tuple (-1, -1, z_prime.shape[-1])
        # this is the desired size of the index!!
        # this is basically saying to repeat the inner ones :) 

        # this is the number of latent variables
        print(z_prime.shape)

        index = self.index.expand((-1, -1, z_prime.shape[-1]))
        #print(index)
        # tensor.expand(which depth to expand, how many to expand)
        dim = z_prime.dim()
        #print(dim)

        # now, we gather z_prime to shape (n_vertex, nu, d_lat) and flatten final two dimensions
        z_prime_unsqueezed = z_prime.unsqueeze(-2)
        print(z_prime_unsqueezed.shape)
        #print(f'z_prime_unsqueezed is {z_prime_unsqueezed}') # add an extra dimension at -2
        #print(f'dim - 1 is {dim-1}') # this is 1
        #print(type((1,))) # tuples concatonate when adding!!
        #print(f'product is {(1, self.nu, 1)}')
        # these are the 5 latent variables at each node
        z_prime_changed = z_prime_unsqueezed.repeat((1, self.nu, 1))
        # print(f'z_prime_changed is {z_prime_changed}')
        # this gives a tensor of shape ndofs, 3, dlat
        # currently, the values do not correctly respond to this right thing.
        # i.e. at each vertex, we have 3 repeating values
        print(z_prime_changed.shape)

        z_tilde = torch.gather(
            z_prime_changed,
            0,
            index,
        )#.flatten(start_dim=-2, end_dim=-1)

        # I want to try to do this step by step

        # this constructs a new array with the same shape as index!!



        print(z_tilde.shape)


        #print(z_tilde.size())

        x_ancil_tilde = x_ancil.transpose(-2, -1)
        #print(x_ancil_tilde)

        #y_tilde = torch.cat((z_tilde, x_ancil_tilde), dim=-1)
        return


d_lat = 5
n_ancil = 3

icomesh = UnitIcosahedralSphereMesh(refinement_level=0)
dualmesh = IcosahedralDualMesh(nref=0)
V = FunctionSpace(icomesh, "CG", 1)  # define the function space

n_dual_vertex = dualmesh.dof_count
#print(f'No dual vertices is {n_dual_vertex}')

n_vertex = V.dim()
#print(f'No vertices of firedrake mesh is {n_vertex}')

z_prime = torch.rand(n_dual_vertex, d_lat)
#print(f'Shape of z_prime is {z_prime.shape}')
x_ancil = torch.rand(n_ancil, n_vertex)
#print(f'Shape of x_ancil is {x_ancil.shape}')

testing_decoder = Decoder(V, dualmesh, 1)
testing_decoder.forward(z_prime, x_ancil)