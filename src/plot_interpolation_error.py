"""Calculate error interpolating onto the dual mesh and back"""

from torch.utils.data import DataLoader
from firedrake import *
import os

import argparse

from neural_pde.datasets import load_hdf5_dataset, show_hdf5_header

parser = argparse.ArgumentParser()

parser.add_argument(
    "--output",
    type=str,
    action="store",
    help="path to output folder",
    default="output",
)

parser.add_argument(
    "--data",
    type=str,
    action="store",
    help="file containing the data",
    default="data/data_test_nref4_0.h5",
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

if not os.path.exists(args.output):
    os.makedirs(args.output)

mesh = UnitIcosahedralSphereMesh(dataset.n_ref)
dualmesh = UnitIcosahedralSphereMesh(1) # number of refinements on the dual mesh
V = FunctionSpace(mesh, "CG", 1)
V_dg = FunctionSpace(dualmesh, "DG", 0)


for j, ((X, t), y_target) in enumerate(iter(dataset)):

    f_input = Function(V, name="input")
    f_input.dat.data[:] = X.detach().numpy()[0, :]

    f_process = Function(V_dg, name="process")
    f_process.interpolate(f_input)

    f_output = Function(V, name="output")
    f_output.interpolate(f_process)

    L2_error = norms.errornorm(f_input, f_output)
    print(L2_error)

    file = VTKFile(os.path.join(args.output, f"firedrake_mesh_output_{j:04d}.pvd"))
    file.write(f_input, f_output)

    file = VTKFile(os.path.join(args.output, f"dual_mesh_output_{j:04d}.pvd"))
    file.write(f_process)


