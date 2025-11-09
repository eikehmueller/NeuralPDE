import numpy as np
import pytest
import torch
from neural_pde.hamiltonian_solver import (
    masked_stepsize,
    SymplecticIntegratorFunction,
    Hamiltonian,
)


@pytest.fixture
def rng():
    _rng = np.random.default_rng(seed=3852157)
    return _rng


class SimpleHamiltonian(Hamiltonian):
    """Simple Hamiltonian where the forcing functions are linear maps"""

    def __init__(self, d_lat, d_ancil, rng=None):
        """Initialise new instance

        :arg d_lat: total dimension of state space
        :arg d_ancil: dimension of ancillary space
        """
        super().__init__(d_lat, d_ancil)
        self.linear_q = torch.nn.Linear(d_lat // 2 + d_ancil, d_lat // 2)
        self.linear_p = torch.nn.Linear(d_lat // 2 + d_ancil, d_lat // 2)
        if rng is not None:
            C_0 = 0.1 / np.sqrt(d_lat // 2 + d_ancil)
            with torch.no_grad():
                for p in self.parameters():
                    p.copy_(
                        torch.tensor(
                            rng.uniform(low=-C_0, high=+C_0, size=p.shape)
                        ).float()
                    )

    def F_q(self, p, xi):
        """Forcing function F_q which determines rate of change of q

        :arg p: momentum vector, d_lat/2-dimensional
        :arg xi: ancillary vector, d_ancil-dimensional
        """
        x = torch.cat((p, xi), dim=-1)
        return torch.sigmoid(self.linear_q(x))

    def F_p(self, q, xi):
        """Forcing function F_p which determines rate of change of p

        :arg q: position vector, d_lat/2-dimensional
        :arg xi: ancillary vector, d_ancil-dimensional
        """
        x = torch.cat((q, xi), dim=-1)
        return torch.sigmoid(self.linear_p(x))


def autograd_solver(hamiltonian, X, t_final, dt):
    """Solve with the autograd function SymplecticIntegratorFunction

    Returns the square loss

    :arg X: initial state (q,p,xi)
    :arg t_final: final time
    :arg dt: timestep size
    """
    hamiltonian.zero_grad()
    F = SymplecticIntegratorFunction.apply
    Y = F(X, hamiltonian, t_final, dt)
    loss = torch.sum(Y**2)
    return loss


def naive_solver(hamiltonian, X, t_final, dt):
    """Naive solver relying on PyTorch's automated differentiation

    Returns the square loss

    :arg X: initial state (q,p,xi), tensor of shape (B,d_lat + d_ancil)
    :arg t_final: final time, can be tensor of shape (B,)
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
    t_q = torch.zeros(1)
    t_p = torch.zeros(1)
    # position half-step
    dt_q = masked_stepsize(t_q, t_final, dt / 2)
    q = q + dt_q * hamiltonian.F_q(p, xi)
    t_q += dt / 2
    while True:
        # momentum-step
        dt_p = masked_stepsize(t_p, t_final, dt)
        p = p + dt_p * hamiltonian.F_p(q, xi)
        t_p += dt
        # position-step
        dt_q = masked_stepsize(t_q, t_final, dt)
        q = q + dt_q * hamiltonian.F_q(p, xi)
        t_q += dt
        if torch.count_nonzero(dt_q) == 0 and torch.count_nonzero(dt_p) == 0:
            break
    loss = torch.sum(q**2) + torch.sum(p**2) + torch.sum(xi**2)
    return loss


def test_hamiltonian_loss_scalar(rng):
    """Check that the loss is computed correctly

    This will verify that the forward operator is correct for the scalar version
    """
    d_lat = 16
    d_ancil = 3
    dt = 0.1
    t_final = rng.uniform(low=3, high=4)
    hamiltonian = SimpleHamiltonian(d_lat, d_ancil, rng)
    X = torch.tensor(rng.normal(0, 1, size=(d_lat + d_ancil))).float()
    X.requires_grad = True
    loss_naive = naive_solver(hamiltonian, X, t_final, dt)
    loss_autograd = autograd_solver(hamiltonian, X, t_final, dt)

    assert np.allclose(loss_autograd.detach(), loss_naive.detach())


def test_hamiltonian_loss_batched(rng):
    """Check that the loss is computed correctly

    This will verify that the forward operator is correct for the batched version
    """
    d_lat = 16
    d_ancil = 3
    batch_size = 4
    dt = 0.1
    t_final = torch.tensor(rng.uniform(low=3, high=4, size=batch_size)).float()
    hamiltonian = SimpleHamiltonian(d_lat, d_ancil, rng)
    X = torch.tensor(rng.normal(0, 1, size=(batch_size, d_lat + d_ancil))).float()
    X.requires_grad = True
    loss_naive = naive_solver(hamiltonian, X, t_final, dt)
    loss_autograd = autograd_solver(hamiltonian, X, t_final, dt)

    assert np.allclose(loss_autograd.detach(), loss_naive.detach())


def test_hamiltonian_input_gradients_scalar(rng):
    """Check that gradients with respect to inputs is computed correctly

    This will verify that the the backward method works as behaved
    """
    d_lat = 16
    d_ancil = 3
    t_final = rng.uniform(low=3, high=4)
    dt = 0.1
    hamiltonian = SimpleHamiltonian(d_lat, d_ancil, rng)
    X = torch.tensor(rng.normal(0, 1, size=(d_lat + d_ancil))).float()
    X.requires_grad = True
    loss = autograd_solver(hamiltonian, X, t_final, dt)
    loss.backward()
    grad_autograd = X.grad
    loss = naive_solver(hamiltonian, X, t_final, dt)
    loss.backward()
    grad_naive = X.grad
    assert np.allclose(grad_autograd, grad_naive)


def test_hamiltonian_input_gradients_batched(rng):
    """Check that gradients with respect to inputs is computed correctly

    This will verify that the the backward method works as behaved
    """
    d_lat = 16
    d_ancil = 3
    batch_size = 4
    dt = 0.1
    t_final = torch.tensor(rng.uniform(low=3, high=4, size=batch_size)).float()
    hamiltonian = SimpleHamiltonian(d_lat, d_ancil, rng)
    X = torch.tensor(rng.normal(0, 1, size=(batch_size, d_lat + d_ancil))).float()
    X.requires_grad = True
    loss = autograd_solver(hamiltonian, X, t_final, dt)
    loss.backward()
    grad_autograd = X.grad
    loss = naive_solver(hamiltonian, X, t_final, dt)
    loss.backward()
    grad_naive = X.grad
    assert np.allclose(grad_autograd, grad_naive)


def test_hamiltonian_parameter_gradients_scalar(rng):
    """Check that gradients with respect to model parameters are computed correctly

    This will verify that the the backward method works as behaved (scalar version)
    """
    d_lat = 16
    d_ancil = 1
    dt = 0.1
    t_final = rng.uniform(low=3, high=4)
    hamiltonian = SimpleHamiltonian(d_lat, d_ancil, rng)
    X = torch.tensor(rng.normal(0, 1, size=(d_lat + d_ancil))).float()
    X.requires_grad = True
    loss = naive_solver(hamiltonian, X, t_final, dt)
    loss.backward()
    grad_naive = []
    for p in hamiltonian.parameters():
        grad_naive.append(p.grad.detach())

    loss = autograd_solver(hamiltonian, X, t_final, dt)
    loss.backward()
    grad_autograd = []
    for p in hamiltonian.parameters():
        grad_autograd.append(p.grad.detach())

    for g1, g2 in zip(grad_autograd, grad_naive):
        assert np.allclose(g1, g2, rtol=1e-4)


def test_hamiltonian_parameter_gradients_batched(rng):
    """Check that gradients with respect to model parameters are computed correctly

    This will verify that the the backward method works as behaved (batched version)
    """
    d_lat = 16
    d_ancil = 3
    batch_size = 4
    dt = 0.01
    t_final = torch.tensor(rng.uniform(low=3, high=4, size=batch_size)).float()
    hamiltonian = SimpleHamiltonian(d_lat, d_ancil, rng)
    X = torch.tensor(rng.normal(0, 1, size=(batch_size, d_lat + d_ancil))).float()
    X.requires_grad = True
    loss = naive_solver(hamiltonian, X, t_final, dt)
    loss.backward()
    grad_naive = []
    for p in hamiltonian.parameters():
        grad_naive.append(p.grad.detach())

    loss = autograd_solver(hamiltonian, X, t_final, dt)
    loss.backward()
    grad_autograd = []
    for p in hamiltonian.parameters():
        grad_autograd.append(p.grad.detach())

    for g1, g2 in zip(grad_autograd, grad_naive):
        assert np.allclose(g1, g2, rtol=1e-3)
