import numpy as np
import torch
from hamiltonian_solver import Solver


class HamiltonianForcing(torch.nn.Module):
    def __init__(self, d_dyn, d_ancil):
        super().__init__()
        self.linear = torch.nn.Linear(d_dyn + d_ancil, d_dyn, bias=True)
        self.d_dyn = d_dyn
        self.d_ancil = d_ancil

    def forward(self, z, xi):
        x = torch.cat((z, xi), dim=-1)
        return self.linear(x)


def models(d_lat, d_ancil):
    F_q = HamiltonianForcing(d_lat // 2, d_ancil)
    F_p = HamiltonianForcing(d_lat // 2, d_ancil)
    return F_q, F_p


def autograd_solver(model_q, model_p, X, n, dt):
    d_lat = 2 * model_q.d_dyn
    d_ancil = model_q.d_ancil
    model_q.zero_grad()
    model_p.zero_grad()
    F = Solver.apply
    Y = F(X, model_q, model_p, d_lat, d_ancil, n, dt)
    loss = torch.sum(Y**2)
    loss.backward()
    return loss


def naive_solver(model_q, model_p, X, n, dt):
    d_lat = 2 * model_q.d_dyn
    d_ancil = model_q.d_ancil
    model_q.zero_grad()
    model_p.zero_grad()
    m = X.shape[-1]
    q0, p0, xi = torch.split(
        X.clone().detach(), [d_lat // 2, d_lat // 2, d_ancil], dim=-1
    )
    q0.requires_grad = True
    p0.requires_grad = True
    xi.requires_grad = True
    q = q0
    p = p0
    q = q + dt / 2 * model_q(p, xi)
    for j in range(n):
        p = p + dt * model_p(q, xi)
        rho = 1 / 2 if j == n - 1 else 1
        q = q + rho * dt * model_q(p, xi)
    loss = torch.sum(q**2) + torch.sum(p**2) + torch.sum(xi**2)
    loss.backward()
    return loss


def test_hamiltonian_loss():
    dt = 0.1
    n_t = 8
    d_lat = 8
    d_ancil = 3
    model_q, model_p = models(d_lat, d_ancil)
    X = torch.tensor(np.arange(d_lat + d_ancil, dtype=np.float32), requires_grad=True)
    loss_autograd = autograd_solver(model_q, model_p, X, n_t, dt)
    loss_naive = naive_solver(model_q, model_p, X, n_t, dt)
    assert np.allclose(loss_autograd.detach(), loss_naive.detach())


def test_hamiltonian_input_gradients():
    dt = 0.1
    n_t = 8
    d_lat = 8
    d_ancil = 3
    model_q, model_p = models(d_lat, d_ancil)
    X = torch.tensor(np.arange(d_lat + d_ancil, dtype=np.float32), requires_grad=True)
    autograd_solver(model_q, model_p, X, n_t, dt)
    grad_autograd = X.grad
    naive_solver(model_q, model_p, X, n_t, dt)
    grad_naive = X.grad
    assert np.allclose(grad_autograd, grad_naive)


def test_hamiltonian_parameter_gradients():
    dt = 0.1
    n_t = 8
    d_lat = 8
    d_ancil = 3
    model_q, model_p = models(d_lat, d_ancil)
    X = torch.tensor(np.arange(d_lat + d_ancil, dtype=np.float32), requires_grad=True)
    autograd_solver(model_q, model_p, X, n_t, dt)
    grad_autograd = []
    for p in list(model_q.parameters()) + list(model_p.parameters()):
        grad_autograd.append(p.grad.detach())

    naive_solver(model_q, model_p, X, n_t, dt)
    grad_naive = []
    for p in list(model_q.parameters()) + list(model_p.parameters()):
        grad_naive.append(p.grad.detach())
    for g1, g2 in zip(grad_autograd, grad_naive):
        assert np.allclose(g1, g2)
