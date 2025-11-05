import numpy as np
import torch
from hamiltonian_solver import Solver
import pytest


def autograd_solver(model_q, model_p, X, n, dt):
    model_q.zero_grad()
    model_p.zero_grad()
    F = Solver.apply
    Y = F(X, model_q, model_p, n, dt)
    loss = torch.sum(Y**2)
    loss.backward()
    return loss


def naive_solver(model_q, model_p, X, n, dt):
    model_q.zero_grad()
    model_p.zero_grad()
    m = X.shape[-1]
    q0, p0 = torch.split(X.clone().detach(), m // 2, dim=-1)
    q0.requires_grad = True
    p0.requires_grad = True
    q = q0
    p = p0
    q = q + dt / 2 * model_q(p)
    for j in range(n):
        p = p + dt * model_p(q)
        rho = 1 / 2 if j == n - 1 else 1
        q = q + rho * dt * model_q(p)
    loss = torch.sum(q**2) + torch.sum(p**2)
    loss.backward()
    return loss


def models(m):
    linear_q = torch.nn.Linear(m // 2, m // 2, bias=True)
    linear_p = torch.nn.Linear(m // 2, m // 2, bias=True)
    return linear_q, linear_p


def test_hamiltonian_loss():
    dt = 0.1
    n = 8
    m = 8
    model_q, model_p = models(m)
    X = torch.tensor(np.arange(m, dtype=np.float32), requires_grad=True)
    loss_autograd = autograd_solver(model_q, model_p, X, n, dt)
    loss_naive = naive_solver(model_q, model_p, X, n, dt)
    assert np.allclose(loss_autograd.detach(), loss_naive.detach())


def test_hamiltonian_input_gradients():
    dt = 0.1
    n = 8
    m = 8
    model_q, model_p = models(m)
    X = torch.tensor(np.arange(m, dtype=np.float32), requires_grad=True)
    autograd_solver(model_q, model_p, X, n, dt)
    grad_autograd = X.grad
    naive_solver(model_q, model_p, X, n, dt)
    grad_naive = X.grad
    assert np.allclose(grad_autograd, grad_naive)


def test_hamiltonian_parameter_gradients():
    dt = 0.1
    n = 8
    m = 8
    model_q, model_p = models(m)
    X = torch.tensor(np.arange(m, dtype=np.float32), requires_grad=True)
    autograd_solver(model_q, model_p, X, n, dt)
    grad_autograd = []
    for p in list(model_q.parameters()) + list(model_p.parameters()):
        grad_autograd.append(p.grad.detach())

    naive_solver(model_q, model_p, X, n, dt)
    grad_naive = []
    for p in list(model_q.parameters()) + list(model_p.parameters()):
        grad_naive.append(p.grad.detach())
    for g1, g2 in zip(grad_autograd, grad_naive):
        assert np.allclose(g1, g2)
