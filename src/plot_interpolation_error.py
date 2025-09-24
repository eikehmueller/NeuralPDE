"""Calculate error interpolating onto the dual mesh and back"""

from torch.utils.data import DataLoader
from firedrake import *
import os

import argparse

from neural_pde.datasets import load_hdf5_dataset, show_hdf5_header
from neural_pde.intergrid import Interpolator, AdjointInterpolator
from neural_pde.icosahedral_dual_mesh import IcosahedralDualMesh
from neural_pde.loss_functions import normalised_rmse

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
V = FunctionSpace(mesh, "CG", 1)
dualmesh = IcosahedralDualMesh(1)

newmesh = V.mesh()

for j, ((X, t), y_target) in enumerate(iter(dataset)):

    f_input = Function(V, name="input")

    interp = Interpolator(fs_from=V, fs_to=dualmesh)
    f_dual = interp.forward(f_input)
    adjinterp = AdjointInterpolator(fs_from=dualmesh, fs_to=V.mesh())
    f_interpolated = adjinterp.forward(f_dual)

    L2_error = normalised_rmse(f_input, f_interpolated)
    print(L2_error)

