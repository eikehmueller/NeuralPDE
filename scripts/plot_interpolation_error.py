"""Calculate error interpolating onto the dual mesh and back"""
import sys
sys.path.insert(0, "/home/katie795/NeuralPDE_workspace/data")
from torch.utils.data import DataLoader
from firedrake import *
import os
from firedrake import norms
import argparse
import tomllib

from neural_pde.datasets import load_hdf5_dataset, show_hdf5_header
from neural_pde.model import load_model
import torch

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    type=str,
    action="store",
    help="name of parameter file",
    default="config.toml",
)

parser.add_argument(
    "--output",
    type=str,
    action="store",
    help="path to output folder",
    default="../results",
)

parser.add_argument(
    "--data_directory",
    type=str,
    action="store",
    help="file containing the data",
    default="../data/",
)

parser.add_argument(
    "--model",
    type=str,
    action="store",
    help="directory containing the trained model",
    default="../saved_model",
)

args, _ = parser.parse_known_args()

with open(args.config, "rb") as f:
    config = tomllib.load(f)

print()
print(f"==== data ====")
print()

show_hdf5_header(f"{args.data_directory}{config["data"]["test"]}")
print()

dataset = load_hdf5_dataset(f"{args.data_directory}{config["data"]["test"]}")
train_ds = load_hdf5_dataset(f"{args.data_directory}{config["data"]["train"]}")
batch_size = len(dataset)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

if not os.path.exists(args.output):
    os.makedirs(args.output)

model = load_model(args.model,  mean=torch.from_numpy(train_ds.mean), std=torch.from_numpy(train_ds.std))

mesh = UnitIcosahedralSphereMesh(dataset.n_ref)
dualmesh = UnitIcosahedralSphereMesh(1) # number of refinements on the dual mesh
V = FunctionSpace(mesh, "CG", 1)
V_dg = FunctionSpace(dualmesh, "DG", 0)

L2_interpolated_error_list = []
L2_learned_error_list = []

j=0
for j, ((X, t), y_target) in enumerate(iter(dataset)):
    X = torch.unsqueeze(X, 0)
    f_input = Function(V, name="input")
    f_input.dat.data[:] = X.detach().numpy()[0, 0, :]

    f_process = Function(V_dg, name="process")
    f_process.interpolate(f_input)

    f_interpolated_output = Function(V, name="interpolated_output")
    f_interpolated_output.interpolate(f_process)

    f_model = model(X, t)
    f_learned_output = Function(V, name="model_output")
    f_learned_output.dat.data[:] = f_model.detach().numpy()[0, 0, :]

    L2_error_interpolated = norms.errornorm(f_input, f_interpolated_output)
    L2_interpolated_error_list.append(L2_error_interpolated)
    L2_error_learned = norms.errornorm(f_input, f_learned_output)
    L2_learned_error_list.append(L2_error_learned)

    file = VTKFile(os.path.join(args.output, f"firedrake_mesh_output_{j:04d}.pvd"))
    file.write(f_input, f_interpolated_output, f_learned_output)

    file = VTKFile(os.path.join(args.output, f"dual_mesh_output_{j:04d}.pvd"))
    file.write(f_process)
    j+=1

print(f'Average interpolated L2 error: {np.average(np.asarray(L2_interpolated_error_list))}')
print(f'Average learned      L2 error: {np.average(np.asarray(L2_learned_error_list))}')

