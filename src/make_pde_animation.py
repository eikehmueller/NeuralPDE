from firedrake import *
import time
import tomllib
import argparse
from neural_pde.datasets import load_hdf5_dataset
parser = argparse.ArgumentParser()

parser.add_argument(
    "--data",
    type=str,
    action="store",
    help="file containing the data",
    default="../data/data_test_nref3_0.h5",
)
args, _ = parser.parse_known_args()
with open(args.config, "rb") as f:
    config = tomllib.load(f)
dataset = load_hdf5_dataset(args.data)
mesh = UnitIcosahedralSphereMesh(dataset.n_ref)
V_DG = FunctionSpace(mesh, "DG", 0)
V = FunctionSpace(mesh, "CG", 1)
animation_file_pde = VTKFile(os.path.join(args.output, f"animation_pde.pvd"))
f_pred_dg = Function(V_DG, name="input")
dt = config["architecture"]["dt"]
t = float(dataset.metadata["t_lowest"]) 
t_final = float(dataset.metadata["t_highest"])
(X, _), __ = next(iter(dataset)) 

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


with open("timing.dat", "w", encoding="utf8") as f:
    print("t,nn,pde", file=f)
    while t < t_final:
        print(f"{t:8.4f},", file=f, end="")
        t_elapsed = t_finish - t_start
        print(f"{t_elapsed:8.4e},", file=f, end="")
        t_start = time.perf_counter()
        f_pred_dg.assign(time_integration(X, V, V_DG, t, dt, dataset.metadata["omega"]))
        t_finish = time.perf_counter()
        t_elapsed = t_finish - t_start
        print(f"{t_elapsed:8.4e}", file=f)
        animation_file_pde.write(f_pred_dg, time=t)
        t += dt
        print(f"time = {t:8.4f}")