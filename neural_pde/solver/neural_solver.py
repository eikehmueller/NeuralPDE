import torch
from neural_pde.solver.hamiltonian import (
    masked_stepsize,
    NearestNeighbourHamiltonian,
)

__all__ = ["ForwardEulerNeuralSolver", "SymplecticNeuralSolver"]


class ForwardEulerNeuralSolver(torch.nn.Module):
    """Neural solver which integrates forward the equations of motion in latent space.

    A state in latent space can be written as [Y^{(k)}_{i,j},a_{i,j}] where i is the index of
    the vertex on the dual mesh.

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
        dual_mesh,
        interaction_model,
        stepsize=1.0,
    ):
        """Initialise new instance

        :arg dual_mesh: the dual mesh that defines the latent state
        :arg interaction_model: model that describes the interactions Phi_theta
        :arg stepsize: size dt of steps
        """
        super().__init__()
        self.dual_mesh = dual_mesh
        self.interaction_model = interaction_model
        self.stepsize = stepsize
        print(f"stepsize is {stepsize}")
        # Construct nested list of the form
        #
        #   [[0,n^0_0,n^0_1,n^0_2,]... [j,n^j_0,n^j_1,n^j_2],...]
        #
        # where n^j_0, n^j_1,n^j_2 are the indices of the three neighbours of
        # each vertex of the dual mesh.
        self._neighbour_list = [
            [j] + beta for j, beta in enumerate(self.dual_mesh.neighbour_list)
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


class SymplecticIntegratorFunction(torch.autograd.Function):
    """Autograd function which realises a symplectic integrator based on Strang splitting

    By implementing the forward and backward methods the construction of the graph can be avoided.
    """

    @staticmethod
    def forward(ctx, x, hamiltonian, t_final, dt):
        """Forward pass, compute state at time T = n_t*dt

        The input x in R^{n_state*(d_lat+d_ancil)} is split as
        x = (q_0,p_0,xi) where q, p are tensors of shape (n_state,d_lat/2) and xi is
        a tensor of shape (n_state,d_ancil). The integrator returns (q_T,p_T,xi)
        where q_T, p_T are obtained by integrating the hamiltonian system to time T = n_t*dt with a
        symplectic integrator based on lowest-order Strang splitting.

        The functions F_q and F_p are of the form

            F_q, F_p: R^{n_state x d_lat/2} x R^{n_state x d_ancil} -> R^{n_state x d_lat/2}

        and they each depend on a set of (learnable) parameters theta

        :arg x: input state of shape (B,d_{lat}+d_{ancil})
        :arg hamiltonian: hamiltonian object which provides forcing functions F_q, F_p
        :arg t_final: final time can be a vector of batchsize
        :arg dt: size of timesteps
        """
        n_state = hamiltonian.n_state
        d_lat = hamiltonian.d_lat
        d_ancil = hamiltonian.d_ancil
        assert (
            x.shape[-1] == d_lat + d_ancil
        ), "Final dimension must be of size d_lat + d_ancil"
        assert x.shape[-2] == n_state, "Penultimate dimension must be of size n_state"
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
