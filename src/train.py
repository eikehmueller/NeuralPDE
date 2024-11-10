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
from neural_pde.model import build_model

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
    "--model",
    type=str,
    action="store",
    help="directory to stored trained model in",
    default="saved_model",
)

args, _ = parser.parse_known_args()

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
    model.parameters(), lr=config["optimiser"]["initial_learning_rate"]
)
gamma = (
    config["optimiser"]["final_learning_rate"]
    / config["optimiser"]["initial_learning_rate"]
) ** (1 / config["optimiser"]["nepoch"])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=gamma)

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
    scheduler.step()

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
    writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], epoch)
    print()
writer.flush()

end = timer()
print(f"Runtime: {timedelta(seconds=end-start)}")

model.save(args.model)
