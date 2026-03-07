"""Various methods for plotting results"""

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
    "--dataset",
    type=str,
    action="store",
    help="whether to use the training, testing, or validation dataset",
    default="test",
)

parser.add_argument(
    "--animate_model",
    action="store_true",
    help="whether to produce an animation using the neural network model",
)

parser.add_argument(
    "--dataset_and_model",
    action="store_true",
    help="whether to produce vtu files for the dataset",
)

parser.add_argument(
    "--animate_dataset",
    action="store_true",
    help="whether to animate the given dataset (default test dataset)",
)

parser.add_argument(
    "--output",
    type=str,
    action="store",
    help="path to output folder",
    default="results/",
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

show_hdf5_header(f"{args.data_directory}{config["data"][args.dataset]}")
print()


if not os.path.exists(args.output):
    os.makedirs(args.output)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = load_hdf5_dataset(f"{args.data_directory}{config["data"][args.dataset]}")

batch_size = config["optimiser"]["batchsize"]
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

train_ds = load_hdf5_dataset(f"{args.data_directory}{config["data"]["train"]}")

overall_mean = torch.mean(torch.from_numpy(train_ds.mean), axis=1)[
    train_ds.n_func_in_dynamic + train_ds.n_func_in_ancillary :
].to(device)

overall_std = torch.mean(torch.from_numpy(train_ds.std), axis=1)[
    train_ds.n_func_in_dynamic + train_ds.n_func_in_ancillary :
].to(device)

mesh = IcosahedralSphereMesh(dataset.radius, dataset.n_ref)
V = FunctionSpace(mesh, "CG", 1)
V_DG = FunctionSpace(mesh, "DG", 0)
dt = config["time"]["dt"]
dt_scaled = torch.tensor(config["time"]["dt"] / 26349.050394344416).float()
t_initial = dataset._t_initial * 26349.050394344416
t_final = dataset._t_final * 26349.050394344416
t_final_max_scaled = torch.tensor(config["time"]["tfinalmax"] / 26349.050394344416)
X_all = torch.tensor(dataset._data)


def find_initial_data():
    # for the animation, we don't need any testing data, only the inital state!

    _, indices = torch.sort(torch.tensor(t_initial))
    sorted_X = torch.tensor(dataset._data[indices, :, :])

    X = torch.zeros(1, 6, len(sorted_X[0,0,:]))
        # input data
    X[0, 0, :] = sorted_X[0, 0, :]  # h data
    X[0, 1, :] = sorted_X[0, 1, :]
    X[0, 2, :] = sorted_X[0, 2, :]

    X[0, 3, :] = sorted_X[0, 3, :]
    X[0, 4, :] = sorted_X[0, 4, :]
    X[0, 5, :] = sorted_X[0, 5, :]
    X = torch.tensor(X, dtype=torch.float32).to(device) 
    return X
    


if args.animate_dataset:
    # TRY THIS WITHOUT SORTING!!
    print("Plotting dataset")
    sorted_t, indices = torch.sort(torch.tensor(t_initial))
    sorted_X = torch.tensor(dataset._data[indices, :, :])
    dataset_file = VTKFile(os.path.join(args.output, "dataset_animation.pvd"))

    for i in range(len(sorted_t)):
        if not(np.isclose(sorted_t[i], sorted_t[i-1], atol=0.1)):
            f_input_d = Function(V, name=f"height")
            f_input_d.dat.data[:] = sorted_X.detach().cpu().numpy()[i, 0, :]
            f_input_div = Function(V, name="divergence")
            f_input_div.dat.data[:] = sorted_X.detach().cpu().numpy()[i, 1, :]
            f_input_vor = Function(V, name="vorticity")
            f_input_vor.dat.data[:] = sorted_X.detach().cpu().numpy()[i, 2, :]

            dataset_file.write(
                f_input_d,
                f_input_div,
                f_input_vor,
                time=sorted_t[i].numpy()
            )
        

if args.animate_model:
    print("Plotting model")
    model, _, _ = load_model(args.model)
    animation_file_nn = VTKFile(os.path.join(args.output, "model_animation.pvd"))
    t = torch.tensor(0.).to(device)
    X = find_initial_data().to(device)
    h_pred = Function(V, name="h")
    div_pred = Function(V, name="div")
    vor_pred = Function(V, name="vor")

    print(f"t_final_max_scaled is {t_final_max_scaled}")
    while t < t_final_max_scaled:
        t = torch.tensor(t).to(device)
        y_pred = model(X, t)

        h_pred.dat.data[:] = y_pred.detach().cpu().numpy()[0, 0, :]
        div_pred.dat.data[:] = y_pred.detach().cpu().numpy()[0, 1, :]
        vor_pred.dat.data[:] = y_pred.detach().cpu().numpy()[0, 2, :]
        #X[0, 0, :] = y_pred[0, 0, :]  # h data
        #X[0, 1, :] = y_pred[0, 1, :]
        #X[0, 2, :] = y_pred[0, 2, :]

        t_actual = 26349.050394344416 * t.detach().cpu().numpy()
        animation_file_nn.write(h_pred, div_pred, vor_pred, time=t_actual)

        t += dt_scaled

if args.dataset_and_model:
    print("Plotting dataset and model")
    model, _, _ = load_model(args.model)

    file = VTKFile(os.path.join(args.output, f"dataset_and_model.pvd"))
    for j, ((X, t), y_target) in enumerate(iter(dataset)):
        X = X.to(device)
        t = t.to(device)
        y_target = y_target.to(device)
        X = torch.unsqueeze(X, 0)
        y_pred = model(X, t)
        y_pred = torch.squeeze(y_pred, 0)
        yt = torch.unsqueeze(y_target, 0)

        X = torch.squeeze(X, 0)

        f_input_d = Function(V, name=f"input_d")
        f_input_d.dat.data[:] = X.detach().cpu().numpy()[0, :]
        f_input_div = Function(V, name="input_div")
        f_input_div.dat.data[:] = X.detach().cpu().numpy()[1, :]
        f_input_vor = Function(V, name="input_vor")
        f_input_vor.dat.data[:] = X.detach().cpu().numpy()[2, :]

        f_target_d = Function(V, name=f"target_d")
        f_target_d.dat.data[:] = y_target.cpu().detach().numpy()[0, :]
        f_target_div = Function(V, name="target_div")
        f_target_div.dat.data[:] = y_target.cpu().detach().numpy()[1, :]
        f_target_vor = Function(V, name="target_vor")
        f_target_vor.dat.data[:] = y_target.cpu().detach().numpy()[2, :]
        f_target = y_target.detach()[0:2, :]

        f_pred_d = Function(V, name=f"pred_d_t")
        f_pred_d.dat.data[:] = y_pred.detach().cpu().numpy()[0, :]
        f_pred_div = Function(V, name="pred_div")
        f_pred_div.dat.data[:] = y_pred.detach().cpu().numpy()[1, :]
        f_pred_vor = Function(V, name="pred_vor")
        f_pred_vor.dat.data[:] = y_pred.detach().cpu().numpy()[2, :]
        f_pred = y_pred.detach()[0:2, :]

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
            time=t_initial[j]
        )
