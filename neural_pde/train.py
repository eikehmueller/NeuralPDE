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
import gc
from neural_pde.data.datasets import load_hdf5_dataset, show_hdf5_header
from neural_pde.util.loss_functions import (
    multivariate_normalised_rmse_with_data as loss_fn,
)
from neural_pde.util.loss_functions import individual_function_rmse as loss_fn2
from neural_pde.util.loss_functions import multivariate_normalised_rmse as loss_fn3
from neural_pde.model.model import build_model, load_model

gc.collect()
torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#torch.set_default_device(device)

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
print("==== parameters ====")
print()
with open(args.config, "r") as f:
    for line in f.readlines():
        print(line.strip())


print()
print("==== data ====")
print()

show_hdf5_header(f"{args.data_directory}{config["data"]["train"]}")
print()
show_hdf5_header(f"{args.data_directory}{config["data"]["valid"]}")
print()

train_ds = load_hdf5_dataset(f"{args.data_directory}{config["data"]["train"]}")
valid_ds = load_hdf5_dataset(f"{args.data_directory}{config["data"]["valid"]}")

overall_mean = torch.mean(torch.from_numpy(train_ds.mean), axis=1)[
    train_ds.n_func_in_dynamic + train_ds.n_func_in_ancillary :
].to(device)
overall_std = torch.mean(torch.from_numpy(train_ds.std), axis=1)[
    train_ds.n_func_in_dynamic + train_ds.n_func_in_ancillary :
].to(device)

train_dl = DataLoader(
    train_ds, batch_size=config["optimiser"]["batchsize"], shuffle=True, drop_last=True
)
valid_dl = DataLoader(
    valid_ds, batch_size=config["optimiser"]["batchsize"], drop_last=True
)

### Tensors for input for model

#n_ref = train_ds.n_ref
#n_func_in_dynamic = train_ds.n_func_in_dynamic
#n_func_in_ancillary = train_ds.n_func_in_ancillary
#n_func_target = train_ds.n_func_target
#arc = config["architecture"]
#mean = torch.from_numpy(train_ds.mean).to(device)
#std = torch.from_numpy(train_ds.std).to(device)
#radius = train_ds.radius


if not ("checkpoint.pt" in os.listdir(args.model)):  # load model or initialise new one
    print("Building model and optimiser")
    model = build_model(
        train_ds.n_ref,
        train_ds.n_func_in_dynamic,
        train_ds.n_func_in_ancillary,
        train_ds.n_func_target,
        config["architecture"],
        mean=torch.from_numpy(train_ds.mean),
        std=torch.from_numpy(train_ds.std),
        radius=train_ds.radius
    )

    model = model.to(device)  # transfer the model to the GPU

    optimiser = torch.optim.Adam(
        model.parameters(), lr=config["optimiser"]["initial_learning_rate"]
    )
    prev_epoch = 0
else:
    print("Loading model and optimiser")
    model, optimiser, prev_epoch = load_model(
        args.model,
    )
    print(f"Previous epoch is {prev_epoch}")

gamma = (
    config["optimiser"]["final_learning_rate"]
    / config["optimiser"]["initial_learning_rate"]
) ** (1 / config["optimiser"]["nepoch"])

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=gamma)

print(f"Running on device {device}")
writer = SummaryWriter("../results/runs", flush_secs=5)
# main training loop
print("STARTING TRAINING LOOP")
for epoch in range(config["optimiser"]["nepoch"]):
    print(
        f"epoch {epoch + 1 + prev_epoch} of {config["optimiser"]["nepoch"] + prev_epoch}"
    )
    train_loss = 0
    function_loss = torch.zeros(train_ds.n_func_target).to(device)
    model.train(True)
    for (Xb, tb), yb in tqdm.tqdm(train_dl):

        Xb = Xb.to(device)
        yb = yb.to(device)
        #tb /= train_ds.timescale
        tb = tb.to(device)
        y_pred = model(Xb, tb)  # make a prediction
        optimiser.zero_grad()  # resets all of the gradients to zero, otherwise the gradients are accumulated
        loss = loss_fn(y_pred, yb, overall_mean, overall_std)  # calculate the loss
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
    percent_loss = 0
    for (Xv, tv), yv in valid_dl:
        Xv = Xv.to(device)  # move to GPU
        yv = yv.to(device)  # move to GPU
        #tv /= train_ds.timescale
        tv = tv.to(device)  # move to GPU
        yv_pred = model(Xv, tv)  # make a prediction
        
        loss = loss_fn(yv_pred, yv, overall_mean, overall_std)  # calculate the loss

        valid_loss += loss.item() / (
            valid_ds.n_samples // config["optimiser"]["batchsize"]
        )
        loss2 = loss_fn2(yv_pred, yv, overall_mean, overall_std)
        function_loss += loss2 / (
            valid_ds.n_samples // config["optimiser"]["batchsize"]
        )

        loss3 = loss_fn3(yv_pred, yv)
        percent_loss += loss3.item() / (
            valid_ds.n_samples // config["optimiser"]["batchsize"]
        )

    print(f"    training loss: {train_loss:8.3e}, validation loss: {valid_loss:8.3e}")
    print(f"    Loss for individual functions: {function_loss}")
    print(f"    Percentage loss: {100 * percent_loss:6.3f} %")
    writer.add_scalars(
        "loss",
        {"train": train_loss, "valid": valid_loss},
        epoch,
    )
    writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], epoch)
    print()
    if epoch % 10 == 0:
        state = {
            "epoch": epoch + prev_epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimiser.state_dict(),
        }
        print("Saving model...")
        model.save(state, args.model)
    torch.cuda.empty_cache()
writer.flush()

end = timer()
print(f"Runtime: {timedelta(seconds=end-start)}")

state = {
    "epoch": epoch + prev_epoch,
    "state_dict": model.state_dict(),
    "optimizer": optimiser.state_dict(),
}

model.save(state, args.model)
