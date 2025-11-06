from abc import ABC, abstractmethod
import numpy as np
import torch


class Hamiltonian(ABC, torch.nn.Module):
    def __init__(self, d_lat, d_ancil):
        super().__init__()
        assert d_lat == 2 * d_lat // 2, "d_lat has to be a multiple of two"
        self.d_lat = d_lat
        self.d_ancil = d_ancil

    @abstractmethod
    def F_q(self, p, xi):
        pass

    @abstractmethod
    def F_p(self, q, xi):
        pass


class Solver(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, hamiltonian, n_t, dt):
        """Forward pass, compute state at time T = n_t*dt

        The input x in R^{d_lat+d_ancil} is split as x = (q_0,p_0,xi) where q, p are vectors
        of length d_lat/2 and xi is a vector of length d_ancil. The function returns (q_T,p_T,xi)
        where q_T, p_T are obtained by integrating to time T = n_t*dt with the symplectic integrator.
        The functions F_q and F_p are of the form

            F_q, F_p: R^{d_lat/2} x R^{d_ancil} -> R^{d_lat/2}

        and they each depend on a set of parameters theta_q, theta_p.

        :arg x: input state of shape (B,d_{lat}+d_{ancil})
        :arg hamiltonian: hamiltonian object which provides function F_q, F_p
        :arg d_lat: dimension of dynamic latent space
        :arg d_ancil: dimension of ancillary space
        """
        d_lat = hamiltonian.d_lat
        d_ancil = hamiltonian.d_ancil
        assert (
            x.shape[-1] == d_lat + d_ancil
        ), "Last dimension must be of size d_lat + d_ancil"
        assert d_lat == 2 * d_lat // 2, "Dimension d_lat as to be a multiple of two"
        q, p, xi = torch.split(
            x.clone().detach(), [d_lat // 2, d_lat // 2, d_ancil], dim=-1
        )
        ctx.hamiltonian = hamiltonian
        ctx.n_t = n_t
        ctx.dt = dt
        with torch.no_grad():
            q = q + dt / 2 * hamiltonian.F_q(p, xi)
            for j in range(n_t):
                p = p + dt * hamiltonian.F_p(q, xi)
                rho = 1 / 2 if j == n_t - 1 else 1
                q = q + rho * dt * hamiltonian.F_q(p, xi)
            output = torch.cat([q, p, xi], dim=-1)
            ctx.save_for_backward(output)
        return output

    @staticmethod
    def _backward_step(z_1, z_2, xi, F, grad_1, grad_2, grad_xi, theta, scaling_factor):
        with torch.no_grad():
            z_1 -= scaling_factor * F(z_2, xi)
        _z_2 = z_2.detach()
        _z_2.requires_grad = True
        with torch.enable_grad():
            dz_1 = F(_z_2, xi)
        (dF_2, dF_xi, *dtheta) = torch.autograd.grad(
            dz_1, [_z_2, xi] + theta, grad_outputs=grad_2, allow_unused=True
        )
        for param, grad_param in zip(theta, dtheta):
            if grad_param is not None:
                if param.grad is None:
                    param.grad = scaling_factor * grad_param
                else:
                    param.grad += scaling_factor * grad_param
        grad_1 += scaling_factor * dF_2
        grad_xi += scaling_factor * dF_xi

    @staticmethod
    def backward(ctx, grad_output):
        (output,) = ctx.saved_tensors
        d_lat = ctx.hamiltonian.d_lat
        d_ancil = ctx.hamiltonian.d_ancil
        F_q = ctx.hamiltonian.F_q
        F_p = ctx.hamiltonian.F_p
        split = [d_lat // 2, d_lat // 2, d_ancil]
        q, p, xi = torch.split(output.clone().detach(), split, dim=-1)
        xi.requires_grad = True
        grad_q, grad_p, grad_xi = torch.split(
            grad_output.clone().detach(), split, dim=-1
        )
        theta = list(ctx.hamiltonian.parameters())
        for j in range(ctx.n_t):
            rho = 1 / 2 if j == 0 else 1
            Solver._backward_step(
                q, p, xi, F_q, grad_p, grad_q, grad_xi, theta, rho * ctx.dt
            )
            Solver._backward_step(p, q, xi, F_p, grad_q, grad_p, grad_xi, theta, ctx.dt)
        Solver._backward_step(q, p, xi, F_q, grad_p, grad_q, grad_xi, theta, ctx.dt / 2)
        grad_input = torch.cat([grad_q, grad_p, grad_xi], dim=-1)
        return grad_input, None, None, None
