"""Evaluated a saved model on a dataset"""

import torch
from torch.utils.data import DataLoader
from firedrake import *
import os
import time

import argparse

from neural_pde.datasets import load_hdf5_dataset, show_hdf5_header
from neural_pde.loss_functions import multivariate_normalised_rmse as metric
from neural_pde.model import load_model


def time_integration(X_initial, V, V_DG, t_final, dt, omega):
    """Integrate PDE using very simple forward Euler method

    :arg X_initial: initial field
    :arg V: DG0 function space
    :arg dt: timestep size
    :arg omega: angular velocity

    """
    mesh = V.mesh()
    n = FacetNormal(mesh)
    q_cg = Function(V)

    phi = TestFunction(V_DG)
    psi = TrialFunction(V_DG)
    q_cg.dat.data[:] = X_initial.detach().numpy()[0, :]
    q = Function(V_DG).interpolate(q_cg)

    a_lhs = phi * psi * dx
    z_hat = as_vector([0, 0, 1])
    x = as_vector(SpatialCoordinate(mesh))

    u_adv = Constant(-float(omega)) * cross(z_hat, x)

    rhs = phi * q * dx + Constant(dt) * (
        q * div(phi * u_adv) * dx
        - conditional(inner(u_adv("+"), n("+") - n("-")) > 0, q("+"), q("-"))
        * (inner(u_adv("+"), n("+")) * phi("+") + inner(u_adv("-"), n("-")) * phi("-"))
        * dS
    )

    q_new = Function(V_DG)
    a_lhs = phi * psi * dx
    lvp = LinearVariationalProblem(a_lhs, rhs, q_new)
    lvs = LinearVariationalSolver(
        lvp, solver_parameters={"ksp_type": "preonly", "pc_type": "jacobi"}
    )

    t = 0
    while t < t_final:
        lvs.solve()
        q.assign(q_new)
        t += dt
    return q


parser = argparse.ArgumentParser()

parser.add_argument(
    "--output",
    type=str,
    action="store",
    help="path to output folder",
    default="output_for_evaluation",
)

parser.add_argument(
    "--model",
    type=str,
    action="store",
    help="directory containing the trained model",
    default="saved_model",
)

parser.add_argument(
    "--data",
    type=str,
    action="store",
    help="file containing the data",
    default="data/data_test_swes_nref_3_10.h5",
)

args, _ = parser.parse_known_args()

print()
print(f"==== data ====")
print()

show_hdf5_header(args.data)
print()

dataset = load_hdf5_dataset(args.data)

batch_size = len(dataset)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

model = load_model(args.model)


# validation
model.train(False)
avg_loss = 0
for (Xv, tv), yv in dataloader:
    yv_pred = model(Xv, tv)
    loss = metric(yv_pred, yv)
    print(dataset.n_samples)
    print(batch_size)
    avg_loss += loss.item() #/ (dataset.n_samples / batch_size)

print(f"average relative error: {100*avg_loss:6.3f} %")

if not os.path.exists(args.output):
    os.makedirs(args.output)


mesh = UnitIcosahedralSphereMesh(dataset.n_ref)
V = FunctionSpace(mesh, "CG", 1)
V_DG = FunctionSpace(mesh, "DG", 0)
(X, _), __ = next(iter(dataset))
dt = 0.1
t = 0.0
t_final = float(dataset.metadata["t_final_max"])
animation_file_nn = VTKFile(os.path.join(args.output, f"animation.pvd"))
animation_file_pde = VTKFile(os.path.join(args.output, f"animation_pde.pvd"))
f_pred = Function(V, name="input")
f_pred_dg = Function(V_DG, name="input")

with open("timing.dat", "w", encoding="utf8") as f:
    print("t,nn,pde", file=f)
    while t < t_final:
        print(f"{t:8.4f},", file=f, end="")
        t_start = time.perf_counter()
        y_pred = model(X, torch.tensor(t))
        t_finish = time.perf_counter()
        t_elapsed = t_finish - t_start
        print(f"{t_elapsed:8.4e},", file=f, end="")
        f_pred.dat.data[:] = y_pred.detach().numpy()[0, :]
        animation_file_nn.write(f_pred, time=t)
        t_start = time.perf_counter()
        f_pred_dg.assign(time_integration(X, V, V_DG, t, dt, dataset.metadata["omega"]))
        t_finish = time.perf_counter()
        t_elapsed = t_finish - t_start
        print(f"{t_elapsed:8.4e}", file=f)
        animation_file_pde.write(f_pred_dg, time=t)
        t += dt
        print(f"time = {t:8.4f}")


for j, ((X, t), y_target) in enumerate(iter(dataset)):
    y_pred = model(X, t)

    f_input = Function(V, name="input")
    f_input.dat.data[:] = X.detach().numpy()[3, :]

    f_target = Function(V, name="target")
    f_target.dat.data[:] = y_target.detach().numpy()[0, :]

    f_pred = Function(V, name="predicted")
    f_pred.dat.data[:] = y_pred.detach().numpy()[0, :]

    file = VTKFile(os.path.join(args.output, f"output_{j:04d}_t{t:6.3f}.pvd"))
    file.write(f_input, f_target, f_pred)
