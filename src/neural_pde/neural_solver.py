import tensorflow as tf
import numpy as np


class NeuralSolver(tf.keras.layers.Layer):
    """Neural solver which integrates forward the equations of motion in latent space.

    A state in latent space can be written as [Y^{(k)}_{i,j},a_{i,j}] where i is the index of
    the vertex on the dual mesh that defines the spherical patch covering.

        Y^{(k)}_{i,j} for j = 0,1,...,d_{latent}-1

    is the latent state vector which evolves and

        a_{i,j} for j = 0,1,...,d_{ancillary}-1

    is the ancillary state vector which remains unchanged.

    Assume that we have some model parametrised by learnable parameters theta with

        Phi_theta([Y_{i,j},a_{i,j}]) = Phi_theta(Y_{i_0,j'},Y_{i_1,j'},Y_{i_2,j'};
                                                 a_{i_0,j'},a_{i_1,j},a_{i_2,j'})

    where {i_0,i_1,i_2} is the set of vertices which contains the vertex i itself and its two
    neighbours. Note that the input shape of Phi_theta is (3,d_{latent}+d_{ancillary}) and the
    output shape is (d_{latent},).

    Then the update implemented here is the forward-Euler model

        Y^{(k+1)}_{i,j} = Y^{(k)}_{i,j} + dt * Phi_theta([Y^{(k)}_{i,j},a_{i,j}])

    """

    def __init__(
        self,
        spherical_patch_covering,
        interaction_model,
        latent_dim,
        nsteps=1,
        stepsize=1.0,
    ):
        """Initialise new instance

        :arg spherical_patch_covering: the spherical patch covering that defines the topology and
            local patches
        :arg interaction_model: model that describes the interactions Phi_theta
        :arg latent_dim: dimension of latent space
        :arg nsteps: number of forward-Euler steps
        :arg stepsize: size dt of steps
        """
        super().__init__()
        self.spherical_patch_covering = spherical_patch_covering
        self.interaction_model = interaction_model
        self.latent_dim = latent_dim
        self.nsteps = nsteps
        self.stepsize = stepsize

        # construct indices for gathering neighbour data in the correct shape
        self._gather_indices = tf.constant(
            [
                [[idx] for idx in subindices]
                for subindices in self.spherical_patch_covering.neighbour_list
            ]
        )

    def call(self, inputs):
        """Carry out a number of forward-Euler steps for the latent variables on the dual mesh

        :arg inputs: tensor of shape (B,n_patch,d_{latent}+d_{ancillary})
        """
        Y = inputs

        # expand indices
        batch_size = inputs.shape[0]
        indices = tf.repeat(
            tf.expand_dims(tf.constant(self._gather_indices), axis=0),
            repeats=batch_size,
            axis=0,
        )

        # work out paddings for adding zeros in the ancillary dimensions
        paddings = np.zeros((3, 2))
        ancillary_dim = inputs.shape[-1] - self.latent_dim
        paddings[-1, 1] = ancillary_dim

        for _ in range(self.nsteps):
            # ---- stage 1 ---- gather to tensor Z of shape
            #                   (B,n_patch,3,d_{latent}+d_{ancillary})
            print("Y.shape = ", Y.shape)
            print("indices = ", indices)
            Z = tf.gather_nd(indices=indices, params=Y, batch_dims=1)

            # ---- stage 2 ---- apply interaction model to obtain tensor of shape
            #                   (B,n_patch,d_{dynamic})
            fZ = tf.stack([self.interaction_model(z) for z in tf.unstack(Z)])
            # ---- stage 3 ---- pad with zeros in last dimension to obtain a tensor dY of shape
            #                   (B,n_patch,d_{dynamic}+d_{ancillary})
            dY = tf.pad(fZ, paddings=paddings, mode="CONSTANT", constant_values=0)

            # ---- stage 4 ---- update Y = Y + dt*dY
            Y += self.stepsize * dY

        return Y
