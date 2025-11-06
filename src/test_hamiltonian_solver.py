import numpy as np
import torch
from hamiltonian_solver import SymplecticIntegratorFunction, Hamiltonian


class LinearHamiltonian(Hamiltonian):
    """Simple Hamiltonian where the forcing functions are linear maps"""

    def __init__(self, d_lat, d_ancil):
        """Initialise new instance

        :arg d_lat: total dimension of state space
        :arg d_ancil: dimension of ancillary space
        """
        super().__init__(d_lat, d_ancil)
        self.linear_q = torch.nn.Linear(d_lat // 2 + d_ancil, d_lat // 2)
        self.linear_p = torch.nn.Linear(d_lat // 2 + d_ancil, d_lat // 2)

    def F_q(self, p, xi):
        """Forcing function F_q which determines rate of change of q

        :arg p: momentum vector, d_lat/2-dimensional
        :arg xi: ancillary vector, d_ancil-dimensional
        """
        x = torch.cat((p, xi), dim=-1)
        return self.linear_q(x)

    def F_p(self, q, xi):
        """Forcing function F_p which determines rate of change of p

        :arg q: position vector, d_lat/2-dimensional
        :arg xi: ancillary vector, d_ancil-dimensional
        """
        x = torch.cat((q, xi), dim=-1)
        return self.linear_p(x)


def autograd_solver(hamiltonian, X, n_t, dt):
    """Solve with the autograd function SymplecticIntegratorFunction

    Returns the square loss

    :arg X: initial state (q,p,xi)
    :arg n_t: number of time steps
    :arg dt: timestep size
    """
    hamiltonian.zero_grad()
    F = SymplecticIntegratorFunction.apply
    Y = F(X, hamiltonian, n_t, dt)
    loss = torch.sum(Y**2)
    loss.backward()
    return loss


def naive_solver(hamiltonian, X, n_t, dt):
    """Naive solver relying on PyTorchs automated differentiation

    Returns the square loss

    :arg X: initial state (q,p,xi)
    :arg n_t: number of time steps
    :arg dt: timestep size
    """
    d_lat = hamiltonian.d_lat
    d_ancil = hamiltonian.d_ancil
    hamiltonian.zero_grad()
    q0, p0, xi = torch.split(
        X.clone().detach(), [d_lat // 2, d_lat // 2, d_ancil], dim=-1
    )
    q0.requires_grad = True
    p0.requires_grad = True
    xi.requires_grad = True
    q = q0
    p = p0
    q = q + dt / 2 * hamiltonian.F_q(p, xi)
    for j in range(n_t):
        p = p + dt * hamiltonian.F_p(q, xi)
        rho = 1 / 2 if j == n_t - 1 else 1
        q = q + rho * dt * hamiltonian.F_q(p, xi)
    loss = torch.sum(q**2) + torch.sum(p**2) + torch.sum(xi**2)
    loss.backward()
    return loss


def test_hamiltonian_loss():
    """Check that the loss is computed correctly

    This will verify that the forward operator is correct
    """
    dt = 0.1
    n_t = 8
    d_lat = 8
    d_ancil = 3
    hamiltonian = LinearHamiltonian(d_lat, d_ancil)
    X = torch.tensor(np.arange(d_lat + d_ancil, dtype=np.float32), requires_grad=True)
    loss_autograd = autograd_solver(hamiltonian, X, n_t, dt)
    loss_naive = naive_solver(hamiltonian, X, n_t, dt)
    assert np.allclose(loss_autograd.detach(), loss_naive.detach())


def test_hamiltonian_input_gradients():
    """Check that gradients with respect to inputs is computed correctly

    This will verify that the the backward method works as behaved
    """
    dt = 0.1
    n_t = 8
    d_lat = 8
    d_ancil = 3
    hamiltonian = LinearHamiltonian(d_lat, d_ancil)
    X = torch.tensor(np.arange(d_lat + d_ancil, dtype=np.float32), requires_grad=True)
    autograd_solver(hamiltonian, X, n_t, dt)
    grad_autograd = X.grad
    naive_solver(hamiltonian, X, n_t, dt)
    grad_naive = X.grad
    assert np.allclose(grad_autograd, grad_naive)


def test_hamiltonian_parameter_gradients():
    """Check that gradients with respect to model parameters are computed correctly

    This will verify that the the backward method works as behaved
    """
    dt = 0.1
    n_t = 8
    d_lat = 8
    d_ancil = 3
    hamiltonian = LinearHamiltonian(d_lat, d_ancil)
    X = torch.tensor(np.arange(d_lat + d_ancil, dtype=np.float32), requires_grad=True)
    autograd_solver(hamiltonian, X, n_t, dt)
    grad_autograd = []
    for p in hamiltonian.parameters():
        grad_autograd.append(p.grad.detach())

    naive_solver(hamiltonian, X, n_t, dt)
    grad_naive = []
    for p in hamiltonian.parameters():
        grad_naive.append(p.grad.detach())
    for g1, g2 in zip(grad_autograd, grad_naive):
        assert np.allclose(g1, g2)
