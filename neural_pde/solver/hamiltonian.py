from abc import ABC, abstractmethod
import numpy as np
import torch

__all__ = ["masked_stepsize", "Hamiltonian", "NearestNeighbourHamiltonian"]


def masked_stepsize(t, t_final, dt):
    """Compute the masked stepsize

    The masked stepsize dt_{masked} is the largest possible positive step size which neither exeeds
    dt nor t_final - t.

    More specifically is defined as

        dt_{masked} = { 0                     if t > t_{final}
                      { min(t - t_{final},dt) if t <= t_{final}

    :arg t: current time, scalar
    :arg t_final: final integration time, can be a vector over batches
    :arg dt: maximum stepsize dt
    """
    # Add another dimension to tensor if it contains more than one element
    _t_final = t_final if torch.is_tensor(t_final) else torch.tensor(t_final)
    assert _t_final.ndim < 2
    if _t_final.ndim == 1:
        _t_final = torch.reshape(_t_final, shape=[-1, 1, 1])
    # Now _t_final is either a scalar tensor or a tensor of shape (batch_size,1,1)
    return torch.minimum(torch.maximum(_t_final - t, torch.tensor(0)), torch.tensor(dt))


class Hamiltonian(ABC, torch.nn.Module):
    """Base class for (separable) parameterised Hamiltonians

    Instances of subclasses of this type can be used by the HamiltonianSolver class.
    Assumes that the parametrised Hamiltonian H(q,p,xi) on the n_state x d_lat-dimensional
    phase space is separable such that the forcing function can be written as

        F_q = F_q(p,xi) = + dH/dp(p,xi)
        F_p = F_q(p,xi) = - dH/dq(q,xi)

    where q, p are n x d_lat/2 dimensional state vectors and xi is a
    n_state x d_ancil-dimensional ancillary state vector.

    We further assume that the Hamiltonian is separable, i.e.

        H(q,p,xi) = H_q(q,xi) + H_p(p,xi)

    The equations of motion are

        dq/dt = + dH/dp = + dH_p(p,xi) = F_q(p,xi)
        dp/dt = - dH/dq = - dH_q(q,xi) = F_p(q,xi)

    """

    def __init__(self, n_state, d_lat, d_ancil):
        """Initialise new instance

        :arg n: size of first dimension
        :arg d_lat: dimension of phase space
        :arg d_ancil: dimension of ancillary space
        """
        super().__init__()
        assert d_lat == 2 * d_lat // 2, "d_lat has to be a multiple of two"
        self.n_state = n_state
        self.d_lat = d_lat
        self.d_ancil = d_ancil

    @abstractmethod
    def H_q(self, q, xi):
        """User-defined position-dependent part of Hamiltonian

        :arg q: position
        :arg xi: ancillary variable
        """
        pass

    @abstractmethod
    def H_p(self, p, xi):
        """User-defined momentum-dependent part of Hamiltonian

        :arg p: momentum
        :arg xi: ancillary variable
        """
        pass

    def F_q(self, p, xi):
        """Forcing function F_q which determines rate of change of q

        :arg p: momentum vector, n_state x d_lat/2-dimensional
        :arg xi: ancillary vector, n_state x d_ancil-dimensional
        """
        p_shape = list(p.shape[:-1])
        p_shape[-1] = 1
        grad_outputs = torch.ones(size=p_shape, device=p.device)
        _p = p.detach()
        _p.requires_grad = True
        with torch.enable_grad():
            _H = self.H_p(_p, xi)
        dH = torch.autograd.grad(_H, _p, grad_outputs=grad_outputs, create_graph=True)
        return dH[0]

    def F_p(self, q, xi):
        """Forcing function F_p which determines rate of change of p

        :arg q: position vector, n_state x d_lat/2-dimensional
        :arg xi: ancillary vector, n_state x d_ancil-dimensional
        """
        q_shape = list(q.shape[:-1])
        q_shape[-1] = 1
        grad_outputs = torch.ones(size=q_shape, device=q.device)
        _q = q.detach()
        _q.requires_grad = True
        with torch.enable_grad():
            _H = self.H_q(_q, xi)
        dH = torch.autograd.grad(_H, _q, grad_outputs=grad_outputs, create_graph=True)
        return -dH[0]


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
