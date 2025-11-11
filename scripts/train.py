import sys
sys.path.insert(0, "/home/katie795/NeuralPDE_workspace/data")

from timeit import default_timer as timer
from datetime import timedelta
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from firedrake import *
from firedrake.adjoint import *
import tqdm
import tomllib
import argparse
import os
import cProfile
from neural_pde.datasets import load_hdf5_dataset, show_hdf5_header
from neural_pde.loss_functions import multivariate_normalised_rmse as loss_fn
from neural_pde.model import build_model, load_model

start = timer()

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
    default="../saved_model",
)

args, _ = parser.parse_known_args()

with open(args.config, "rb") as f:
    config = tomllib.load(f)
print()
print("==== parameters ====")
print()
with open(args.config, "r") as f:
    for line in f.readlines():
        print(line.strip())


print()
print("==== data ====")
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

if not os.listdir(args.model): # load model or initialise new one
    print('Building model')
    model = build_model(
        train_ds.n_ref,
        train_ds.n_func_in_dynamic,
        train_ds.n_func_in_ancillary,
        train_ds.n_func_target,
        config["architecture"],
    )
else:
    print('Loading model')
    model = load_model(args.model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # transfer the model to the GPU
print(f"Running on device {device}")

optimiser = torch.optim.Adam(
    model.parameters(), lr=config["optimiser"]["initial_learning_rate"]
) # is this the best learning rate?
gamma = (
    config["optimiser"]["final_learning_rate"]
    / config["optimiser"]["initial_learning_rate"]
) ** (1 / config["optimiser"]["nepoch"])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=gamma)

writer = SummaryWriter('../results/runs', flush_secs=5)
writer.add_text("Model parameters", f"Refinements of dual space: {config["architecture"]["dual_ref"]}\n \
No. radial points per patch: {config["architecture"]["n_radial"]} \n \
Dimension of dynamic latent space: {config["architecture"]["latent_dynamic_dim"]} \n \
Dimension of ancillary latent space: {config["architecture"]["latent_ancillary_dim"]} \n \
Time-step: {config["architecture"]["dt"]}")


# main training loop
for epoch in range(config["optimiser"]["nepoch"]):
    print(f"epoch {epoch + 1} of {config["optimiser"]["nepoch"]}")
    train_loss = 0
    model.train(True)
    for (Xb, tb), yb in tqdm.tqdm(train_dl):
        Xb = Xb.to(device)  # move to GPU
        tb = tb.to(device)  # move to GPU
        yb = yb.to(device)  # move to GPU
        y_pred = model(Xb, tb)  # make a prediction
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
    for (Xv, tv), yv in valid_dl:
        Xv = Xv.to(device)  # move to GPU
        tv = tv.to(device)  # move to GPU
        yv = yv.to(device)  # move to GPU
        yv_pred = model(Xv, tv)  # make a prediction
        loss = loss_fn(yv_pred, yv)  # calculate the loss
        valid_loss += loss.item() / (
            valid_ds.n_samples // config["optimiser"]["batchsize"]
        )
    #cProfile.run('model(Xv, tv)', sort='cumtime')

    print(f"    training loss: {train_loss:8.3e}, validation loss: {valid_loss:8.3e}")
    writer.add_scalars(
        "loss",
        {"train": train_loss, "valid": valid_loss},
        epoch,
    )
    writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], epoch)
    print()
    if epoch % 100 == 0:
        print("Saving model...")
        model.save(args.model)
writer.flush()

end = timer()
print(f"Runtime: {timedelta(seconds=end-start)}")
#print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
#prof.export_chrome_trace("trace.json")
model.save(args.model)
