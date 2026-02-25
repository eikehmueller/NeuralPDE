"""Evaluate a saved model on the test dataset"""

import torch
from torch.utils.data import DataLoader
from firedrake import *
import tomllib
import argparse
import numpy as np
from neural_pde.data.datasets import load_hdf5_dataset, show_hdf5_header
from neural_pde.util.loss_functions import (
    multivariate_normalised_rmse_with_data as metric,
)
from neural_pde.model.model import load_model

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