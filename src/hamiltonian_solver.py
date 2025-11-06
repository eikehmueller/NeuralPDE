import numpy as np
import torch


class Solver(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, F_q, F_p, d_lat, d_ancil, n_t, dt):
        """Forward pass, compute state at time T = n_t*dt

        The input x in R^{d_lat+d_ancil} is split as x = (q_0,p_0,xi) where q, p are vectors
        of length d_lat/2 and xi is a vector of length d_ancil. The function returns (q_T,p_T,xi)
        where q_T, p_T are obtained by integrating to time T = n_t*dt with the symplectic integrator.
        The functions F_q and F_p are of the form

            F_q, F_p: R^{d_lat/2} x R^{d_ancil} -> R^{d_lat/2}

        and they each depend on a set of parameters theta_q, theta_p.

        :arg x: input state of shape (B,d_{lat}+d_{ancil})
        :arg F_q: forcing function for q-part of state vector
        :arg F_p: forcing function for p-part of state vector
        :arg d_lat: dimension of dynamic latent space
        :arg d_ancil: dimension of ancillary space
        """
        assert (
            x.shape[-1] == d_lat + d_ancil
        ), "Last dimension must be of size d_lat + d_ancil"
        assert d_lat == 2 * d_lat // 2, "Dimension d_lat as to be a multiple of two"
        q, p, xi = torch.split(
            x.clone().detach(), [d_lat // 2, d_lat // 2, d_ancil], dim=-1
        )
        ctx.F_q = F_q
        ctx.F_p = F_p
        ctx.d_lat = d_lat
        ctx.d_ancil = d_ancil
        ctx.n_t = n_t
        ctx.dt = dt
        with torch.no_grad():
            q = q + dt / 2 * F_q(p, xi)
            for j in range(n_t):
                p = p + dt * F_p(q, xi)
                rho = 1 / 2 if j == n_t - 1 else 1
                q = q + rho * dt * F_q(p, xi)
            output = torch.cat([q, p, xi], dim=-1)
            ctx.save_for_backward(output)
        return output

    @staticmethod
    def _backward_step(
        z_1, z_2, xi, F, grad_1, grad_2, grad_xi, theta_2, scaling_factor
    ):
        with torch.no_grad():
            z_1 -= scaling_factor * F(z_2, xi)
        _z_2 = z_2.detach()
        _z_2.requires_grad = True
        with torch.enable_grad():
            dz_1 = F(_z_2, xi)
        (dF_2, dF_xi, *dtheta_2) = torch.autograd.grad(
            dz_1, [_z_2, xi] + theta_2, grad_outputs=grad_2
        )
        for x, y in zip(theta_2, dtheta_2):
            if x.grad is None:
                x.grad = scaling_factor * y
            else:
                x.grad += scaling_factor * y
        grad_1 += scaling_factor * dF_2
        grad_xi += scaling_factor * dF_xi

    @staticmethod
    def backward(ctx, grad_output):
        (output,) = ctx.saved_tensors
        split = [ctx.d_lat // 2, ctx.d_lat // 2, ctx.d_ancil]
        q, p, xi = torch.split(output.clone().detach(), split, dim=-1)
        xi.requires_grad = True
        grad_q, grad_p, grad_xi = torch.split(
            grad_output.clone().detach(), split, dim=-1
        )
        theta_q = list(ctx.F_q.parameters())
        theta_p = list(ctx.F_p.parameters())
        for j in range(ctx.n_t):
            rho = 1 / 2 if j == 0 else 1
            Solver._backward_step(
                q, p, xi, ctx.F_q, grad_p, grad_q, grad_xi, theta_q, rho * ctx.dt
            )
            Solver._backward_step(
                p, q, xi, ctx.F_p, grad_q, grad_p, grad_xi, theta_p, ctx.dt
            )
        Solver._backward_step(
            q, p, xi, ctx.F_q, grad_p, grad_q, grad_xi, theta_q, ctx.dt / 2
        )
        grad_input = torch.cat([grad_q, grad_p, grad_xi], dim=-1)
        return grad_input, None, None, None, None, None, None
