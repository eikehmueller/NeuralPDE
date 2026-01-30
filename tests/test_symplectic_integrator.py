import functools
import numpy as np
import scipy as sp
import pytest
import torch
from neural_pde.model.hamiltonian import masked_stepsize, Hamiltonian
from neural_pde.model.neural_solver import SymplecticIntegratorFunction


@pytest.fixture
def rng():
    """Random number generator with fixed seed to guarantee reproducibility"""
    _rng = np.random.default_rng(seed=3852157)
    return _rng


class SimpleHamiltonian(Hamiltonian):
    """Simple Hamiltonian used for testing"""

    def __init__(self, n_state, d_lat, d_ancil, rng=None, dtype=torch.float):
        """Initialise new instance

        :arg n_state: number of states
        :arg d_lat: total dimension of state space
        :arg d_ancil: dimension of ancillary space
        :arg rng: random number generator to use for initialising the weights
        :arg dtype: data type to use (set to torch.double for testing)
        """
        super().__init__(n_state, d_lat, d_ancil)
        self.linear_q = torch.nn.Linear(
            n_state * (d_lat // 2 + d_ancil), 1, dtype=dtype
        )
        self.linear_p = torch.nn.Linear(
            n_state * (d_lat // 2 + d_ancil), 1, dtype=dtype
        )
        if rng is not None:
            C_0 = 0.1 / np.sqrt(n_state * (d_lat // 2 + d_ancil))
            with torch.no_grad():
                for p in self.parameters():
                    p.copy_(
                        torch.tensor(rng.uniform(low=-C_0, high=+C_0, size=p.shape)).to(
                            dtype
                        )
                    )

    def H_q(self, q, xi):
        """Position-dependent part of Hamiltonian

        :arg q: position vector, d_lat/2-dimensional
        :arg xi: ancillary vector, d_ancil-dimensional
        """
        x = torch.cat((q, xi), dim=-1)
        return (
            1
            / 2
            * torch.sum(
                torch.sigmoid(self.linear_q(torch.flatten(x, start_dim=-2, end_dim=-1)))
                ** 2,
                dim=-1,
                keepdim=True,
            )
        )

    def H_p(self, p, xi):
        """Momentum-dependent part of Hamiltonian

        :arg p: momentum vector, d_lat/2-dimensional
        :arg xi: ancillary vector, d_ancil-dimensional
        """
        x = torch.cat((p, xi), dim=-1)
        return (
            1
            / 2
            * torch.sum(
                torch.sigmoid(self.linear_p(torch.flatten(x, start_dim=-2, end_dim=-1)))
                ** 2,
                dim=-1,
                keepdim=True,
            )
        )


class QuadraticHamiltonian(Hamiltonian):
    """Quadratic Hamiltonian of the form H = 1/2 p^T M p + 1/2 q^T V q

    The matrices M and V are constructed such that they are SPD
    """

    def __init__(self, n_state, d_lat, d_ancil, rng=None, dtype=torch.float):
        """Initialise new instance

        :arg n_state: number of states
        :arg d_lat: size of latent dimension
        :arg d_ancil: size of ancillary dimension (unused)
        :arg rng: random number generator to ensure reproducibility
        :arg dtype: datatype, set to torch.double for testing
        """
        super().__init__(n_state, d_lat, d_ancil)
        A = rng.normal(size=(d_lat // 2, d_lat // 2))
        B = rng.normal(size=(d_lat // 2, d_lat // 2))
        self.M = torch.tensor(np.eye(d_lat // 2) + A.T @ A / d_lat).to(dtype)
        self.V = torch.tensor(np.eye(d_lat // 2) + B.T @ B / d_lat).to(dtype)

    def H_q(self, q, xi):
        """Position-dependent part of Hamiltonian

        returns 1/2 sum_a q_a^T V q_a where q_a is the position of the a-th state

        :arg q: position vector, d_lat/2-dimensional
        :arg xi: ancillary vector, d_ancil-dimensional
        """
        return (
            1 / 2 * torch.einsum("...ai,ij,...aj->...", q, self.V, q).unsqueeze(dim=-1)
        )

    def H_p(self, p, xi):
        """Momentum-dependent part of Hamiltonian

        returns 1/2 sum_a p_a^T M p_a where p_a is the momentum of the a-th state

        :arg p: momentum vector, d_lat/2-dimensional
        :arg xi: ancillary vector, d_ancil-dimensional
        """
        return (
            1 / 2 * torch.einsum("...ai,ij,...aj->...", p, self.M, p).unsqueeze(dim=-1)
        )

    def exact_solution(self, x_0, t):
        """Compute exact solution after given time

        Starting with the initial state x_0 = (q_0,p_0,xi) compute the solution
        x(t) = (q(t),p(t),xi) at some later time t. This solution is given by

        (q(t),p(t))^T = exp(t*[[0,M],[-V,0]]) (q_0,p_0)^T

        :arg x_0: initial state (q_0,p_0)
        :arg t: final time
        """
        # Construct matrix to exponentiate
        A = np.zeros(shape=(self.d_lat, self.d_lat))
        A[: self.d_lat // 2, self.d_lat // 2 :] = +self.M.detach().numpy()
        A[self.d_lat // 2 :, : self.d_lat // 2] = -self.V.detach().numpy()
        qp_0, xi = torch.split(x_0, [self.d_lat, self.d_ancil], dim=-1)
        qp_0 = qp_0.detach().numpy()
        A = A.reshape((x_0.ndim - 1) * [1] + [self.d_lat, self.d_lat])
        if torch.is_tensor(t):
            t = t.reshape(t.shape + x_0.ndim * (1,)).numpy()
        exp_A = sp.linalg.expm(t * A)
        qp = torch.tensor(np.einsum("...ij,...j->...i", exp_A, qp_0))
        return torch.cat((qp, xi), dim=-1)


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


def test_loss_scalar(rng):
    """Check that the loss is computed correctly

    This will verify that the forward operator is correct for the scalar version
    """
    n_state = 2
    d_lat = 16
    d_ancil = 3
    dt = 0.1
    t_final = rng.uniform(low=3, high=4)
    hamiltonian = SimpleHamiltonian(n_state, d_lat, d_ancil, rng)
    X = torch.tensor(rng.normal(0, 1, size=(n_state, d_lat + d_ancil))).float()
    X.requires_grad = True
    loss_naive = naive_solver(hamiltonian, X, t_final, dt)
    loss_autograd = autograd_solver(hamiltonian, X, t_final, dt)

    assert np.allclose(loss_autograd.detach(), loss_naive.detach())


def test_loss_batched(rng):
    """Check that the loss is computed correctly

    This will verify that the forward operator is correct for the batched version
    """
    n_state = 2
    d_lat = 16
    d_ancil = 3
    batch_size = 4
    dt = 0.1
    t_final = torch.tensor(rng.uniform(low=3, high=4, size=batch_size)).float()
    hamiltonian = SimpleHamiltonian(n_state, d_lat, d_ancil, rng)
    X = torch.tensor(
        rng.normal(0, 1, size=(batch_size, n_state, d_lat + d_ancil))
    ).float()
    X.requires_grad = True
    loss_naive = naive_solver(hamiltonian, X, t_final, dt)
    loss_autograd = autograd_solver(hamiltonian, X, t_final, dt)

    assert np.allclose(loss_autograd.detach(), loss_naive.detach())


def test_input_gradients_scalar(rng):
    """Check that gradients with respect to inputs is computed correctly

    This will verify that the the backward method works as behaved
    """
    n_state = 2
    d_lat = 16
    d_ancil = 3
    t_final = rng.uniform(low=3, high=4)
    dt = 0.1
    hamiltonian = SimpleHamiltonian(n_state, d_lat, d_ancil, rng)
    X = torch.tensor(rng.normal(0, 1, size=(n_state, d_lat + d_ancil))).float()
    X.requires_grad = True
    loss = autograd_solver(hamiltonian, X, t_final, dt)
    loss.backward()
    grad_autograd = X.grad
    loss = naive_solver(hamiltonian, X, t_final, dt)
    loss.backward()
    grad_naive = X.grad
    assert np.allclose(grad_autograd, grad_naive)


def test_input_gradients_batched(rng):
    """Check that gradients with respect to inputs is computed correctly

    This will verify that the the backward method works as behaved
    """
    n_state = 2
    d_lat = 16
    d_ancil = 3
    batch_size = 4
    dt = 0.1
    t_final = torch.tensor(rng.uniform(low=3, high=4, size=batch_size)).float()
    hamiltonian = SimpleHamiltonian(n_state, d_lat, d_ancil, rng)
    X = torch.tensor(
        rng.normal(0, 1, size=(batch_size, n_state, d_lat + d_ancil))
    ).float()
    X.requires_grad = True
    loss = autograd_solver(hamiltonian, X, t_final, dt)
    loss.backward()
    grad_autograd = X.grad
    loss = naive_solver(hamiltonian, X, t_final, dt)
    loss.backward()
    grad_naive = X.grad
    assert np.allclose(grad_autograd, grad_naive)


def test_input_gradients_gradcheck(rng):
    """Use gradcheck to compare gradients to finite differences"""
    n_state = 2
    d_lat = 10
    d_ancil = 3
    batch_size = 1
    dt = 0.1
    t_final = torch.tensor(rng.uniform(low=3, high=4, size=batch_size))
    hamiltonian = SimpleHamiltonian(n_state, d_lat, d_ancil, rng, dtype=torch.double)
    X = torch.tensor(rng.normal(0, 1, size=(batch_size, n_state, d_lat + d_ancil)))
    X.requires_grad = True
    Phi = functools.partial(autograd_solver, hamiltonian, t_final=t_final, dt=dt)
    torch.autograd.gradcheck(Phi, X)


def test_parameter_gradients_scalar(rng):
    """Check that gradients with respect to model parameters are computed correctly

    This will verify that the the backward method works as behaved (scalar version)
    """
    n_state = 2
    d_lat = 16
    d_ancil = 1
    dt = 0.1
    t_final = rng.uniform(low=3, high=4)
    hamiltonian = SimpleHamiltonian(n_state, d_lat, d_ancil, rng)
    X = torch.tensor(rng.normal(0, 1, size=(n_state, d_lat + d_ancil))).float()
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
        assert np.allclose(g1, g2)


def test_parameter_gradients_batched(rng):
    """Check that gradients with respect to model parameters are computed correctly

    This will verify that the the backward method works as behaved (batched version)
    """
    n_state = 2
    d_lat = 16
    d_ancil = 3
    batch_size = 4
    dt = 0.01
    t_final = torch.tensor(rng.uniform(low=3, high=4, size=batch_size)).float()
    hamiltonian = SimpleHamiltonian(n_state, d_lat, d_ancil, rng)
    X = torch.tensor(
        rng.normal(0, 1, size=(batch_size, n_state, d_lat + d_ancil))
    ).float()
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
        assert np.allclose(g1, g2)


def test_accuracy(rng):
    """Compare solution to exact solution"""
    n_state = 2
    d_lat = 16
    d_ancil = 3
    dt = 0.001
    t_final = rng.uniform(low=3, high=4)
    hamiltonian = QuadraticHamiltonian(n_state, d_lat, d_ancil, rng, dtype=torch.double)
    X_0 = torch.tensor(
        rng.normal(
            0,
            1,
            size=(
                n_state,
                d_lat + d_ancil,
            ),
        )
    )
    X_exact = hamiltonian.exact_solution(X_0, t_final)
    F = SymplecticIntegratorFunction.apply
    X = F(X_0, hamiltonian, t_final, dt)
    print(X - X_exact)
    print(np.linalg.norm(X - X_exact))
    assert np.allclose(X, X_exact, rtol=1e-4)


def test_convergence(rng):
    """Check that solution converges quadratically"""
    n_state = 2
    d_lat = 16
    d_ancil = 3
    batch_size = 32
    t_final = torch.tensor(rng.uniform(low=3, high=4, size=batch_size))
    hamiltonian = QuadraticHamiltonian(n_state, d_lat, d_ancil, rng, dtype=torch.double)
    X_0 = torch.tensor(rng.normal(0, 1, size=(batch_size, n_state, d_lat + d_ancil)))

    error = []
    for dt in (0.2, 0.1, 0.05, 0.025, 0.0125):
        X_exact = hamiltonian.exact_solution(X_0, t_final)
        F = SymplecticIntegratorFunction.apply
        X = F(X_0, hamiltonian, t_final, dt)
        error.append(torch.max(X - X_exact).detach().numpy())
    rate_empirical = np.asarray([error[j - 1] / error[j] for j in range(1, len(error))])
    print(rate_empirical)
    assert np.all((3.6 < rate_empirical) & (rate_empirical < 4.2))
