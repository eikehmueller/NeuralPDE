import numpy as np
import torch


class Solver(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, F_q, F_p, n, dt):
        m = input.shape[-1]
        q, p = torch.split(input.clone().detach(), m // 2, dim=-1)
        ctx.F_q = F_q
        ctx.F_p = F_p
        ctx.n = n
        ctx.dt = dt
        with torch.no_grad():
            q = q + dt / 2 * F_q(p)
            for j in range(n):
                p = p + dt * F_p(q)
                rho = 1 / 2 if j == n - 1 else 1
                q = q + rho * dt * F_q(p)
            output = torch.cat([q, p], dim=-1)
            ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (output,) = ctx.saved_tensors
        m = output.shape[-1]
        q, p = torch.split(output.clone().detach(), m // 2, dim=-1)
        grad_q, grad_p = torch.split(grad_output.clone().detach(), m // 2, dim=-1)
        theta_q = list(ctx.F_q.parameters())
        theta_p = list(ctx.F_p.parameters())
        for j in range(ctx.n - 1, -1, -1):
            rho = 1 / 2 if j == ctx.n - 1 else 1
            with torch.no_grad():
                q = q - rho * ctx.dt * ctx.F_q(p)
            z_p = p.detach()
            z_p.requires_grad = True
            with torch.enable_grad():
                dq = ctx.F_q(z_p)
            (dJ_q, *dtheta_q) = torch.autograd.grad(
                dq, [z_p] + theta_q, grad_outputs=grad_q
            )
            for x, y in zip(theta_q, dtheta_q):
                if x.grad is None:
                    x.grad = rho * ctx.dt * y
                else:
                    x.grad += rho * ctx.dt * y
            grad_p = grad_p + rho * ctx.dt * dJ_q
            with torch.no_grad():
                p = p - ctx.dt * ctx.F_p(q)
            z_q = q.detach()
            z_q.requires_grad = True
            with torch.enable_grad():
                dp = ctx.F_p(z_q)
            (dJ_p, *dtheta_p) = torch.autograd.grad(
                dp, [z_q] + theta_p, grad_outputs=grad_p
            )
            for x, y in zip(theta_p, dtheta_p):
                if x.grad is None:
                    x.grad = ctx.dt * y
                else:
                    x.grad += ctx.dt * y
            grad_q = grad_q + ctx.dt * dJ_p
        with torch.no_grad():
            q = q - 1 / 2 * ctx.dt * ctx.F_q(p)
        z_p = p.detach()
        z_p.requires_grad = True
        with torch.enable_grad():
            dp = ctx.F_q(z_p)
        (dJ_q, *dtheta_q) = torch.autograd.grad(
            dp, [z_p] + theta_q, grad_outputs=grad_q
        )
        for x, y in zip(theta_q, dtheta_q):
            if x.grad is None:
                x.grad = ctx.dt / 2 * y
            else:
                x.grad += ctx.dt / 2 * y
        grad_p = grad_p + ctx.dt / 2 * dJ_q
        grad_input = torch.cat([grad_q, grad_p], dim=-1)
        return grad_input, None, None, None, None


dt = 0.1
m = 4
n = 8
linear_q = torch.nn.Linear(m // 2, m // 2, bias=True)
linear_p = torch.nn.Linear(m // 2, m // 2, bias=True)

X = torch.tensor(np.arange(m, dtype=np.float32), requires_grad=True)
F = Solver.apply
Y = F(X, linear_q, linear_p, n, dt)
loss = torch.sum(Y**2)
loss.backward()
print(loss, X.grad)

for p in list(linear_q.parameters()) + list(linear_p.parameters()):
    print("dtheta/dp = ", p.grad)

linear_q.zero_grad()
linear_p.zero_grad()
q0, p0 = torch.split(X.clone().detach(), m // 2, dim=-1)
q0.requires_grad = True
p0.requires_grad = True
q = q0
p = p0
q = q + dt / 2 * linear_q(p)
for j in range(n):
    p = p + dt * linear_p(q)
    rho = 1 / 2 if j == n - 1 else 1
    q = q + rho * dt * linear_q(p)
loss2 = torch.sum(q**2) + torch.sum(p**2)
loss2.backward()
print(loss2, q0.grad, p0.grad)
for p in list(linear_q.parameters()) + list(linear_p.parameters()):
    print("dtheta/dp = ", p.grad)
