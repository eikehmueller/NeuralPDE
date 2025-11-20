import numpy as np
import torch
from neural_pde.hamiltonian_solver import Hamiltonian, SymplecticIntegratorFunction

__all__ = ["ForwardEulerNeuralSolver", "SymplecticNeuralSolver"]


class ForwardEulerNeuralSolver(torch.nn.Module):
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

    which comes from

    dY/dt = Phi_theta

    and Phi_theta is a learnable function.
    """

    def __init__(
        self,
        spherical_patch_covering,
        interaction_model,
        stepsize=1.0,
    ):
        """Initialise new instance

        :arg spherical_patch_covering: the spherical patch covering that defines the topology and
            local patches
        :arg interaction_model: model that describes the interactions Phi_theta
        :arg stepsize: size dt of steps
        """
        super().__init__()
        self.spherical_patch_covering = spherical_patch_covering
        self.interaction_model = interaction_model
        self.stepsize = stepsize
        # Construct nested list of the form
        #
        #   [[0,n^0_0,n^0_1,n^0_2,]... [j,n^j_0,n^j_1,n^j_2],...]
        #
        # where n^j_0, n^j_1,n^j_2 are the indices of the three neighbours of
        # each vertex of the dual mesh.
        self._neighbour_list = [
            [j] + beta
            for j, beta in enumerate(self.spherical_patch_covering.neighbour_list)
        ]
        self.register_buffer("index", torch.tensor(self._neighbour_list).unsqueeze(-1))

    def forward(self, x, t_final):
        """Carry out a number of forward-Euler steps for the latent variables on the dual mesh

        :arg x: tensor of shape (B,n_patch,d_{latent}+d_{ancillary}) or (n_patch,d_{latent}+d_{ancillary})
        :arg t_final: final time of the integration, tensor of shape (B,) or scalar
        """
        index = self.index.expand(x.shape[:-2] + (-1, -1, x.shape[-1]))
        dim = x.dim()
        t = 0
        while True:
            # The masked stepsize dt_{masked} is defined as
            #
            #  dt_{masked} = { 0                     if t > t_{final}
            #                { min(t - t_{final},dt) if t <= t_{final}
            #
            masked_stepsize = torch.minimum(
                torch.maximum(t_final - t, torch.tensor(0)), torch.tensor(self.stepsize)
            ).view(t_final.ndim * [-1] + [1, 1])
            # Stop if t > t_{final} for all elements in the batch
            if torch.count_nonzero(masked_stepsize) == 0:
                break
            # input x is of shape (B,n_patch,d_{lat}^{dynamic}+d_{lat}^{ancillary})
            #
            # ---- stage 1 ---- gather to tensor Z of shape
            #                   (B,n_patch,4,d_{lat}^{dynamic}+d_{lat}^{ancillary})

            z = torch.gather(
                x.unsqueeze(-2).repeat((dim - 1) * (1,) + (4, 1)),
                dim - 2,
                index,
            )

            # ---- stage 2 ---- apply interaction model to obtain tensor of shape
            #                   (B,n_patch,d_{lat}^{dynamic})
            fz = self.interaction_model(z)

            # ---- stage 3 ---- pad with zeros in last dimension to obtain a tensor dY of shape
            #                   (B,n_patch,d_{lat}^{dynamic}+d_{lat}^{ancillary})
            dx = torch.nn.functional.pad(
                fz, (0, x.shape[-1] - fz.shape[-1]), mode="constant", value=0
            )
            # ---- stage 4 ---- update Y = Y + dt*dY
            x += masked_stepsize * dx
            t += self.stepsize

        return x


class NearestNeighbourHamiltonian(Hamiltonian):
    """Hamiltonian describing nearest neighbour interactions

    The Hamiltonian is assumed to be of the form

        H(p,q;xi) = sum_{i~j} H_p^{(local)}(p_i,p_j;xi_i,xi_j)
                  + sum_{i~j} H_q^{(local)}(q_i,q_j;xi_i,xi_j)

    where the sums extend over all pairs (i,j) of nearest neighbours.

    H_p^{(local)} and H_q^{(local)} are functions

        R^{d_lat + 2*(d_ancil}) -> R

    """

    def __init__(self, dual_mesh, d_lat, d_ancil, H_q_local, H_p_local):
        """Initialise new instance

        :arg dual_mesh: dual mesh which is used to work out pairs of nearest neighbours
        :arg d_lat: dimension of dynamic part of latent space
        :arg d_ancil: dimension of ancillary latent space
        :arg H_q_local: position dependent part of interaction Hamiltonian H_p^{(local)}
        :arg H_p_local: momentum dependent part of interaction Hamiltonian H_p^{(local)}
        """
        super().__init__(len(dual_mesh.vertices), d_lat, d_ancil)
        self.H_q_local = H_q_local
        self.H_p_local = H_p_local
        # Work out indices of interaction pairs
        edges = {
            tuple(sorted([i, j]))
            for i, beta in enumerate(dual_mesh.neighbour_list)
            for j in beta
        }
        interaction_pairs = np.asarray([list(edge) for edge in edges]).T
        self.register_buffer("index", torch.tensor(interaction_pairs))

    def H_q(self, q, xi):
        """Position-dependent part of Hamiltonian

        :arg q: position, tensor of shape (B,n_patch,d_lat/2)
        :arg xi: ancillary variable, tensor of shape (B,n_patch,d_ancil)
        """
        # transform into four tensors of shapes
        #   (B,n_edges,d_lat/2), (B,n_edges,d_lat/2),
        #   (B,n_edges,d_ancil), (B,n_edges,d_ancil)
        q_i = q[..., self.index[0, :], :]
        q_j = q[..., self.index[1, :], :]
        xi_i = xi[..., self.index[0, :], :]
        xi_j = xi[..., self.index[1, :], :]
        return torch.sum(
            self.H_q_local(torch.cat((q_i, q_j, xi_i, xi_j), dim=-1)), axis=-2
        )

    def H_p(self, p, xi):
        """Momentum-dependent part of Hamiltonian

        :arg p: momentum, tensor of shape (B,n_patch,d_lat/2)
        :arg xi: ancillary variable tensor of shape (B,n_patch,d_ancil)
        """
        p_i = p[..., self.index[0, :], :]
        p_j = p[..., self.index[1, :], :]
        xi_i = xi[..., self.index[0, :], :]
        xi_j = xi[..., self.index[1, :], :]
        return torch.sum(
            self.H_p_local(torch.cat((p_i, p_j, xi_i, xi_j), dim=-1)), axis=-2
        )


class SymplecticNeuralSolver(torch.nn.Module):
    """Neural solver which integrates the equations in latent space"""

    def __init__(self, dual_mesh, d_lat, d_ancil, H_q_local, H_p_local, stepsize=1.0):
        """Initialise new instance

        :arg dual_mesh: dual mesh which is used to work out pairs of nearest neighbours
        :arg d_lat: dimension of dynamic part of latent space
        :arg d_ancil: dimension of ancillary latent space
        :arg H_q_local: position dependent part of interaction Hamiltonian H_p^{(local)}
        :arg H_p_local: momentum dependent part of interaction Hamiltonian H_p^{(local)}
        :arg stepsize: timestep size
        """

        super().__init__()
        self.hamiltonian = NearestNeighbourHamiltonian(
            dual_mesh, d_lat, d_ancil, H_q_local, H_p_local
        )
        self.dt = stepsize

    def forward(self, x, t_final):
        """Carry out a number of symplectic steps for the latent variables on the dual mesh

        :arg x: tensor of shape (B,n_patch,d_{latent}+d_{ancillary}) or (n_patch,d_{latent}+d_{ancillary})
        :arg t_final: final time of the integration, tensor of shape (B,) or scalar
        """
        return SymplecticIntegratorFunction.apply(x, self.hamiltonian, t_final, self.dt)
