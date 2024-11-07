from timeit import default_timer as timer
from datetime import timedelta

start = timer()

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from firedrake import *
from firedrake.adjoint import *
import tqdm
import tomllib
import argparse

from neural_pde.datasets import load_hdf5_dataset, show_hdf5_header
from neural_pde.loss_functions import normalised_mse as loss_fn

from model import build_model

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
    "--path_to_output_folder",
    type=str,
    action="store",
    help="path to output folder",
    default="output",
)

args, _ = parser.parse_known_args()
path_to_output = args.path_to_output_folder

with open(args.config, "rb") as f:
    config = tomllib.load(f)
print()
print(f"==== parameters ====")
print()
with open(args.config, "r") as f:
    for line in f.readlines():
        print(line.strip())


print()
print(f"==== data ====")
print()

show_hdf5_header(config["data"]["train"])
print()
show_hdf5_header(config["data"]["valid"])
print()
train_ds = load_hdf5_dataset(config["data"]["train"])
valid_ds = load_hdf5_dataset(config["data"]["valid"])

train_dl = DataLoader(
    train_ds, batch_size=config["optimiser"]["batchsize"], shuffle=True, drop_last=True
)
valid_dl = DataLoader(
    valid_ds, batch_size=config["optimiser"]["batchsize"], drop_last=True
)

model = build_model(
    train_ds.n_ref,
    train_ds.n_func_in_dynamic,
    train_ds.n_func_in_ancillary,
    train_ds.n_func_target,
    config["architecture"],
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # transfer the model to the GPU
print(f"Running on device {device}")

optimiser = torch.optim.Adam(
    model.parameters(), lr=config["optimiser"]["learning_rate"]
)

writer = SummaryWriter(flush_secs=5)
# main training loop
for epoch in range(config["optimiser"]["nepoch"]):
    print(f"epoch {epoch + 1} of {config["optimiser"]["nepoch"]}")
    train_loss = 0
    model.train(True)
    for Xb, yb in tqdm.tqdm(train_dl):
        Xb = Xb.to(device)  # move to GPU
        yb = yb.to(device)  # move to GPU
        y_pred = model(Xb)  # make a prediction
        optimiser.zero_grad()  # resets all of the gradients to zero, otherwise the gradients are accumulated
        loss = loss_fn(y_pred, yb)  # calculate the loss
        loss.backward()  # take the backwards gradient
        optimiser.step()  # adjust the parameters by the gradient collected in the backwards pass
        # data collection for the model
        train_loss += loss.item() / (
            train_ds.n_samples // config["optimiser"]["batchsize"]
        )

    # validation
    model.train(False)
    valid_loss = 0
    for Xv, yv in valid_dl:
        Xv = Xv.to(device)  # move to GPU
        yv = yv.to(device)  # move to GPU
        yv_pred = model(Xv)  # make a prediction
        loss = loss_fn(yv_pred, yv)  # calculate the loss
        valid_loss += loss.item() / (
            valid_ds.n_samples // config["optimiser"]["batchsize"]
        )

    print(f"    training loss: {train_loss:8.3e}, validation loss: {valid_loss:8.3e}")
    writer.add_scalars(
        "loss",
        {"train": train_loss, "valid": valid_loss},
        epoch,
    )
    print()
writer.flush()

end = timer()
print(f"Runtime: {timedelta(seconds=end-start)}")

# visualise the validation dataset
host_model = model.cpu()
mesh = UnitIcosahedralSphereMesh(train_ds.n_ref)
V = FunctionSpace(mesh, "CG", 1)
for j, (X, y_target) in enumerate(iter(valid_ds)):
    y_pred = host_model(X)

    f_input = Function(V, name="input")
    f_input.dat.data[:] = X.detach().numpy()[0, :]

    f_target = Function(V, name="target")
    f_target.dat.data[:] = y_target.detach().numpy()[0, :]

    f_pred = Function(V, name="predicted")
    f_pred.dat.data[:] = y_pred.detach().numpy()[0, :]

    file = VTKFile(f"{path_to_output}/output_{j:04d}.pvd")
    file.write(f_input, f_target, f_pred)
