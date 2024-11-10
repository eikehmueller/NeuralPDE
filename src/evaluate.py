"""Evaluated a saved model on a dataset"""

from torch.utils.data import DataLoader
from firedrake import *
import os

import argparse

from neural_pde.datasets import load_hdf5_dataset, show_hdf5_header
from neural_pde.loss_functions import normalised_mse as loss_fn
from neural_pde.model import load_model

parser = argparse.ArgumentParser()

parser.add_argument(
    "--output",
    type=str,
    action="store",
    help="path to output folder",
    default="output",
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
    default="data/data_test.h5",
)

args, _ = parser.parse_known_args()

print()
print(f"==== data ====")
print()

show_hdf5_header(args.data)
print()

dataset = load_hdf5_dataset(args.data)

dataloader = DataLoader(dataset, batch_size=1)

model = load_model(args.model)


# validation
model.train(False)
avg_loss = 0
for Xv, yv in dataloader:
    yv_pred = model(Xv)
    loss = loss_fn(yv_pred, yv)
    avg_loss += loss.item() / dataset.n_samples

print(f"average loss: {avg_loss:8.3e}")

if not os.path.exists(args.output):
    os.makedirs(args.output)

mesh = UnitIcosahedralSphereMesh(dataset.n_ref)
V = FunctionSpace(mesh, "CG", 1)
for j, (X, y_target) in enumerate(iter(dataset)):
    y_pred = model(X)

    f_input = Function(V, name="input")
    f_input.dat.data[:] = X.detach().numpy()[0, :]

    f_target = Function(V, name="target")
    f_target.dat.data[:] = y_target.detach().numpy()[0, :]

    f_pred = Function(V, name="predicted")
    f_pred.dat.data[:] = y_pred.detach().numpy()[0, :]

    file = VTKFile(os.path.join(args.output, f"output_{j:04d}.pvd"))
    file.write(f_input, f_target, f_pred)
