from abc import ABC, abstractmethod
import torch

__all__ = ["masked_stepsize", "Hamiltonian", "SymplecticIntegratorFunction"]


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
        _t_final = torch.unsqueeze(_t_final, dim=-1)
    # Now _t_final is either a scalar tensor or a tensor of shape (batch_size,1)
    return torch.minimum(torch.maximum(_t_final - t, torch.tensor(0)), torch.tensor(dt))


class Hamiltonian(ABC, torch.nn.Module):
    """Base class for (separable) parameterised Hamiltonians

    Instances of subclasses of this type can be used by the HamiltonianSolver class.
    Assumes that the parametrised Hamiltonian H(q,p,xi) on the d_lat-dimensional phase space is
    separable such that the forcing function can be written as

        F_q = F_q(p,xi) = + dH/dp(p,xi)
        F_p = p_q(p,xi) = - dH/dq(q,xi)

    where q, p are d_lat/2 dimensional state vectors and xi is a d_ancil-dimensional ancillary state
    vector.

    The equations of motion are

        dq/dt = + dH/dp = F_q
        dp/dt = - dH/dq = F_p

    """

    def __init__(self, d_lat, d_ancil):
        """Initialise new instance

        :arg d_lat: dimension of phase space
        :arg d_ancil: dimension of ancillary space
        """
        super().__init__()
        assert d_lat == 2 * d_lat // 2, "d_lat has to be a multiple of two"
        self.d_lat = d_lat
        self.d_ancil = d_ancil

    @abstractmethod
    def F_q(self, p, xi):
        """Forcing function F_q which determines rate of change of q

        :arg p: momentum vector, d_lat/2-dimensional
        :arg xi: ancillary vector, d_ancil-dimensional
        """
        pass

    @abstractmethod
    def F_p(self, q, xi):
        """Forcing function F_p which determines rate of change of p

        :arg q: position vector, d_lat/2-dimensional
        :arg xi: ancillary vector, d_ancil-dimensional
        """
        pass


class SymplecticIntegratorFunction(torch.autograd.Function):
    """Autograd function which realises a symplectic integrator based on Strang splitting

    By implementing the forward and backward methods the construction of the graph can be avoided.
    """

    @staticmethod
    def forward(ctx, x, hamiltonian, t_final, dt):
        """Forward pass, compute state at time T = n_t*dt

        The input x in R^{d_lat+d_ancil} is split as x = (q_0,p_0,xi) where q, p are vectors
        of length d_lat/2 and xi is a vector of length d_ancil. The integrator returns (q_T,p_T,xi)
        where q_T, p_T are obtained by integrating the hamiltonian system to time T = n_t*dt with a
        symplectic integrator based on lowest-order Strang splitting.

        The functions F_q and F_p are of the form

            F_q, F_p: R^{d_lat/2} x R^{d_ancil} -> R^{d_lat/2}

        and they each depend on a set of (learnable) parameters theta

        :arg x: input state of shape (B,d_{lat}+d_{ancil})
        :arg hamiltonian: hamiltonian object which provides forcing functions F_q, F_p
        :arg t_final: final time can be a vector of batchsize
        :arg dt: size of timesteps
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
        ctx.dt = dt
        ctx.t_final = t_final
        t_q = 0
        t_p = 0
        with torch.no_grad():
            k = 0
            while True:
                # first position update is a half-step
                rho = 1 / 2 if k == 0 else 1
                # position update
                dt_q = masked_stepsize(t_q, t_final, rho * dt)
                q += dt_q * hamiltonian.F_q(p, xi)
                t_q += rho * dt
                # momentum update
                dt_p = masked_stepsize(t_p, t_final, dt)
                p += dt_p * hamiltonian.F_p(q, xi)
                t_p += dt
                if torch.count_nonzero(dt_q) == 0 and torch.count_nonzero(dt_p) == 0:
                    break
                k += 1
            output = torch.cat([q, p, xi], dim=-1)
            ctx.save_for_backward(output)
            ctx.t_q = t_q
            ctx.t_p = t_p
        return output

    @staticmethod
    def _backward_step(z_1, z_2, xi, F, grad_1, grad_2, grad_xi, theta, stepsize):
        """Perform a single update step for either position or velocity

        :arg z_1: q or p
        :arg z_2: p or q
        :arg xi: ancillary variable xi
        :arg F: forcing function to use
        :arg grad_1: first gradient, d/dp or d/dq
        :arg grad_2: second gradition, d/dq or d/dp
        :arg grad_xi: gradient with respect to xi
        :arg theta: learnable parameters of the Hamiltonian model
        :arg stepsize: time step size, scaling C for update

        The two possible updates are:

            If (z_1,z_2) = (q,p), F = F_q, (grad_1,grad_2) = (d/dp,d/dq)

                1. Set q = q - C * F_q(p,xi)
                2. For the variables v = {p,xi,theta}

                    d/dv = d/dv + C*dF_q/dv(p,xi)

            If (z_1,z_2) = (p,q), F = F_p, (grad_1,grad_2) = (d/dq,d/dp)

                1. Set p = p - C * F_p(q,xi)
                2. For the variables v = {q,xi,theta}

                    d/dv = d/dv + C*dF_q/dv(p,xi)


        """
        with torch.no_grad():
            z_1 -= stepsize * F(z_2, xi)
        _z_2 = z_2.detach()
        _z_2.requires_grad = True
        _xi = xi.detach()
        _xi.requires_grad = True
        with torch.enable_grad():
            dz_1 = stepsize * F(_z_2, _xi)
        (dF_2, dF_xi, *dtheta) = torch.autograd.grad(
            dz_1,
            [_z_2, _xi] + theta,
            grad_outputs=grad_2,
            is_grads_batched=False,
            materialize_grads=True,
        )
        for param, grad_param in zip(theta, dtheta):
            if param.grad is None:
                param.grad = grad_param
            else:
                param.grad += grad_param
        grad_1 += dF_2
        grad_xi += dF_xi

    @staticmethod
    def backward(ctx, grad_output):
        """Back-propagate gradients with respect to inputs and model parameters

        :arg grad_output: gradient with respect to output state
        """
        (output,) = ctx.saved_tensors
        d_lat = ctx.hamiltonian.d_lat
        d_ancil = ctx.hamiltonian.d_ancil
        F_q = ctx.hamiltonian.F_q
        F_p = ctx.hamiltonian.F_p
        split = [d_lat // 2, d_lat // 2, d_ancil]
        q, p, xi = torch.split(output.clone().detach(), split, dim=-1)
        grad_q, grad_p, grad_xi = torch.split(
            grad_output.clone().detach(), split, dim=-1
        )
        theta = list(ctx.hamiltonian.parameters())
        t_q = ctx.t_q
        t_p = ctx.t_p
        t_final = ctx.t_final
        while t_q > 0 or t_p > 0:
            # momentum update
            t_p_old = t_p
            t_p = max(t_p_old - ctx.dt, 0)
            dt_p = masked_stepsize(t_p, t_final, t_p_old - t_p)
            SymplecticIntegratorFunction._backward_step(
                p, q, xi, F_p, grad_q, grad_p, grad_xi, theta, dt_p
            )
            t_q_old = t_q
            t_q = max(t_q_old - ctx.dt, 0)
            dt_q = masked_stepsize(t_q, t_final, t_q_old - t_q)
            # position update
            SymplecticIntegratorFunction._backward_step(
                q, p, xi, F_q, grad_p, grad_q, grad_xi, theta, dt_q
            )

        grad_input = torch.cat([grad_q, grad_p, grad_xi], dim=-1)
        return grad_input, None, None, None
