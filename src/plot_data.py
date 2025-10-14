"""Visualise some parts of the dataset from the hdf5 file"""

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
    default="output_for_visualisation",
)

parser.add_argument(
    "--data",
    type=str,
    action="store",
    help="file containing the data",
    default="data/data_valid_swes_nref_3_0.0001.h5",
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
V = FunctionSpace(mesh, "CG", 1)
for j, ((X, t), y_target) in enumerate(iter(dataset)):

    f_input_d = Function(V, name="input_d")
    f_input_d.dat.data[:] = X.detach().numpy()[3, :]
    f_input_u1 = Function(V, name="input_u1")
    f_input_u1.dat.data[:] = X.detach().numpy()[4, :]
    f_input_u2 = Function(V, name="input_u2")
    f_input_u2.dat.data[:] = X.detach().numpy()[5, :]
    f_input_u3 = Function(V, name="input_u3")
    f_input_u3.dat.data[:] = X.detach().numpy()[6, :]

    f_target_d = Function(V, name="target_d")
    f_target_d.dat.data[:] = y_target.detach().numpy()[0, :]
    f_target_u1 = Function(V, name="target_u1")
    f_target_u1.dat.data[:] = y_target.detach().numpy()[1, :]
    f_target_u2 = Function(V, name="target_u2")
    f_target_u2.dat.data[:] = y_target.detach().numpy()[2, :]
    f_target_u3 = Function(V, name="target_u3")
    f_target_u3.dat.data[:] = y_target.detach().numpy()[3, :]

    file = VTKFile(os.path.join(args.output, f"output_{j:04d}.pvd"))
    file.write(f_input_d, f_input_u1, f_input_u2, f_input_u3, f_target_d, f_target_u1, f_target_u2, f_target_u3)
