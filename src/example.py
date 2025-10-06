
from torch import nn
import torch
import tqdm
# define the model - this is a very basic neural network - it doesn't know anything about the PDE
class NeuralNetwork(nn.Module):
    n_points = 100 # number of data points in the x domain of the pde
    n_outputs = 1 # this is the epsilon
    def __init__(self):
        super().__init__()
        self.n_points = 100 # number of data points in the x domain of the pde
        self.n_outputs = 1 # this is the epsilon
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_points, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_outputs),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork() # initialise the model
optimiser = torch.optim.Adam(model.parameters(), lr=0.001) # use standard adams optimiser
batch_size = 32 # number of items to process at once
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=1)

train_dl = 1 # THIS IS A DATALOADER
train_ds = 1 # THIS IS DATASET

valid_dl = 1 # THIS IS ANOTHER DATALOADER
valid_ds = 1 # THIS IS ANOTHER DATASET

def loss_fn(observed, actual):
    # this is where you calculate the error between the data and the PDE solution
    # this is the part I'm not sure about yet, whether this is u or epsilon
    # let me think about this
    return

for epoch in range(32): # loop through the data
    print(f"epoch {epoch + 1}")
    train_loss = 0
    model.train(True)
    for (Xb, tb), yb in tqdm.tqdm(train_dl):
        y_pred = model(Xb, tb)  # make a prediction
        optimiser.zero_grad()  # resets all of the gradients to zero, otherwise the gradients are accumulated
        loss = loss_fn(y_pred, yb)  # calculate the loss
        loss.backward()  # take the backwards gradient
        optimiser.step()  # adjust the parameters by the gradient collected in the backwards pass
        # data collection for the model
        train_loss += loss.item() / (
            train_ds.n_samples // batch_size
        )
    scheduler.step()

    # validation
    model.train(False)
    valid_loss = 0
    for (Xv, tv), yv in valid_dl:
        yv_pred = model(Xv, tv)  # make a prediction
        loss = loss_fn(yv_pred, yv)  # calculate the loss
        valid_loss += loss.item() / (
            valid_ds.n_samples // batch_size
        )

    print(f"    training loss: {train_loss:8.3e}, validation loss: {valid_loss:8.3e}")
    print()
    if epoch % 100 == 0:
        print("Saving model...")
        model.save()