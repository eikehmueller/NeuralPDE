"""Evaluate a saved model on a the test dataset"""

import torch
from torch.utils.data import DataLoader
from firedrake import *
import os
import tomllib
import argparse
import numpy as np
from neural_pde.util.diagnostics import Diagnostics
from neural_pde.data.datasets import load_hdf5_dataset, show_hdf5_header
from neural_pde.util.velocity_functions import Projector as Proj
from neural_pde.util.loss_functions import (
    multivariate_normalised_rmse_with_data as metric,
)
from neural_pde.util.loss_functions import individual_function_rmse as metric2
from neural_pde.model.model import load_model
import matplotlib.pyplot as plt

# Create argparse arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    type=str,
    action="store",
    help="name of parameter file",
    default="config.toml",
)

parser.add_argument(
    "--animate",
    action="store_true",
    help="whether to produce an animation using the neural network model",
)

parser.add_argument(
    "--plot_dataset_and_model",
    action="store_true",
    help="whether to produce vtu files for the dataset",
)

parser.add_argument(
    "--output",
    type=str,
    action="store",
    help="path to output folder",
    default="../results/output_for_evaluation",
)

parser.add_argument(
    "--model",
    type=str,
    action="store",
    help="directory containing the trained model",
    default="../saved_model",
)

parser.add_argument(
    "--data_directory",
    type=str,
    action="store",
    help="directory where the data is saved",
    default="../data/",
)

args, _ = parser.parse_known_args()

with open(args.config, "rb") as f:
    config = tomllib.load(f)

print()
print("==== data ====")
print()

show_hdf5_header(f"{args.data_directory}{config["data"]["test"]}")
print()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = load_hdf5_dataset(f"{args.data_directory}{config["data"]["test"]}")

batch_size = len(dataset)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
train_ds = load_hdf5_dataset(f"{args.data_directory}{config["data"]["train"]}")
overall_mean = torch.mean(torch.from_numpy(train_ds.mean), axis=1)[
    train_ds.n_func_in_dynamic + train_ds.n_func_in_ancillary :
]
overall_std = torch.mean(torch.from_numpy(train_ds.std), axis=1)[
    train_ds.n_func_in_dynamic + train_ds.n_func_in_ancillary :
]

model, _, _ = load_model(args.model)

# validation
model.train(False)
avg_loss = 0
individual_loss = np.zeros(3)
for (Xv, tv), yv in dataloader:
    Xv = Xv.to(device)
    tv = tv.to(device)
    yv = yv.to(device)
    yv_pred = model(Xv, tv)
    loss = metric(yv_pred, yv, overall_mean, overall_std)
    avg_loss += loss.item() / (dataset.n_samples / batch_size)

print(f"average relative error: {100 * avg_loss:6.3f} %")

if not os.path.exists(args.output):
    os.makedirs(args.output)

mesh = UnitIcosahedralSphereMesh(dataset.n_ref)
V = FunctionSpace(mesh, "CG", 1)
V_DG = FunctionSpace(mesh, "DG", 0)
(X, _), __ = next(iter(dataset))
dt = config["architecture"]["dt"] / 26349.050394344416
t_initial = dataset._t_initial / 26349.050394344416
t_final = dataset._t_final / 26349.050394344416
animation_file_nn = VTKFile(os.path.join(args.output, "animation.pvd"))
h_pred = Function(V, name="h")
div_pred = Function(V, name="div")
vor_pred = Function(V, name="vor")

# for the animation, we don't need any testing data, only the inital state!
with CheckpointFile("results/gusto_output/chkpt.h5", "r") as afile:

    mesh_h5 = afile.load_mesh("IcosahedralMesh")
    x, y, z = SpatialCoordinate(mesh_h5)  # spatial coordinate
    V_DG = FunctionSpace(mesh_h5, "DG", 1)
    V_CG = FunctionSpace(mesh_h5, "CG", 1)
    V_BDM = FunctionSpace(mesh_h5, "BDM", 2)
    x_fun = Function(V_CG).interpolate(x)  # collect data on x,y,z coordinates
    y_fun = Function(V_CG).interpolate(y)
    z_fun = Function(V_CG).interpolate(z)

    h_inp = Function(V_CG)  # input function for h
    w1 = afile.load_function(mesh_h5, "u", idx=0)
    h1 = afile.load_function(mesh_h5, "D", idx=0)
    p2 = Proj(V_DG, V_CG)
    p2.apply(h1, h_inp)
    diagnostics = Diagnostics(V_BDM, V_CG)
    vorticity_inp = diagnostics.vorticity(w1)
    divergence_inp = diagnostics.divergence(w1)
    X = np.zeros((1, 6, mesh_h5.num_vertices()), dtype=np.float64)
    # input data
    X[0, 0, :] = h_inp.dat.data  # h data
    X[0, 1, :] = divergence_inp.dat.data
    X[0, 2, :] = vorticity_inp.dat.data

    X[0, 3, :] = x_fun.dat.data  # x coord data
    X[0, 4, :] = y_fun.dat.data  # y coord data
    X[0, 5, :] = z_fun.dat.data  # z coord data
    X = torch.tensor(X, dtype=torch.float32)

if args.animate:
    t = 0
    while t < 20:
        y_pred = model(X, torch.tensor(t))

        h_pred.dat.data[:] = y_pred.detach().numpy()[0, 0, :]
        div_pred.dat.data[:] = y_pred.detach().numpy()[0, 1, :]
        vor_pred.dat.data[:] = y_pred.detach().numpy()[0, 2, :]
        animation_file_nn.write(h_pred, div_pred, vor_pred, time=t)

        t += dt

        print(f"time = {t:8.4f}")

if args.plot_dataset_and_model:
    print("Plotting dataset and model")
    fig, ax = plt.subplots()
    for j, ((X, t), y_target) in enumerate(iter(dataset)):
        X = X.to(device)
        t = t.to(device)
        y_target = y_target.to(device)
        X = torch.unsqueeze(X, 0)
        y_pred = model(X, t)
        y_pred = torch.squeeze(y_pred, 0)
        yt = torch.unsqueeze(y_target, 0)
        print(f"Individual errors are {metric2(y_pred, yt, overall_mean, overall_std)}")
        X = torch.squeeze(X, 0)

        f_input_d = Function(V, name=f"input_d")
        f_input_d.dat.data[:] = X.detach().numpy()[0, :]
        f_input_div = Function(V, name="input_div")
        f_input_div.dat.data[:] = X.detach().numpy()[1, :]
        f_input_vor = Function(V, name="input_vor")
        f_input_vor.dat.data[:] = X.detach().numpy()[2, :]

        f_target_d = Function(V, name=f"target_d_t={t:6.3f}")
        f_target_d.dat.data[:] = y_target.detach().numpy()[0, :]
        f_target_div = Function(V, name="target_div")
        f_target_div.dat.data[:] = y_target.detach().numpy()[1, :]
        f_target_vor = Function(V, name="target_vor")
        f_target_vor.dat.data[:] = y_target.detach().numpy()[2, :]
        f_target = y_target.detach()[0:2, :]

        f_pred_d = Function(V, name=f"pred_d_t={t:6.3f}")
        f_pred_d.dat.data[:] = y_pred.detach().numpy()[0, :]
        f_pred_div = Function(V, name="pred_div")
        f_pred_div.dat.data[:] = y_pred.detach().numpy()[1, :]
        f_pred_vor = Function(V, name="pred_vor")
        f_pred_vor.dat.data[:] = y_pred.detach().numpy()[2, :]
        f_pred = y_pred.detach()[0:2, :]

        file = VTKFile(
            os.path.join(args.output, f"dataset/output_t={t_initial[j]}_{j:04d}.pvd")
        )
        file.write(
            f_input_d,
            f_input_div,
            f_input_vor,
            f_target_d,
            f_target_div,
            f_target_vor,
            f_pred_d,
            f_pred_div,
            f_pred_vor,
        )
        # file.write(f_input_d, f_target_d, f_pred_d,)
        # f_target1 = torch.unsqueeze(f_target, 0)
        # f_pred1 = torch.unsqueeze(f_pred, 0)

        # ax.plot(t, metric(f_target1, f_pred1, overall_mean, overall_std), color="black")
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel("Model RMSE")
    ax.set_title("Total model error")
    plt.tight_layout()
    plt.savefig("../results/model_RMSE_over_time.png")
