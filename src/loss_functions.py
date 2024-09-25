## Module containing our handmade loss functions
from firedrake import *
from firedrake.adjoint import *
import torch
from firedrake.ml.pytorch import fem_operator



def rough_L2_error(y_pred, yb):
    # area of an element in a unit icosahedral mesh
    # This should be at a maximum 2 * h_squared (calculated from a gamma distribution)
    h_squared = (4 * np.pi) / (20 * (4.0 ** num_ref))
    loss = torch.mean(torch.sum((y_pred - yb)**2  * h_squared))
    return loss

def normalised_L2_error(y_pred, yb):
    # length of an edge in a unit icosahedral mesh
    loss = torch.mean(
        torch.sum(torch.sum((y_pred - yb)**2, dim=(1, 2))) 
        / torch.sum(torch.sum((yb)**2, dim=(1, 2)))
        )
    return loss

def firedrake_L2_error(u, w):
   return assemble((u - w) ** 2 * dx)**0.5

def firedrake_loss(y_pred, yb):
    u = Function(V)
    w = Function(V)
    with set_working_tape() as _:
      F = ReducedFunctional(firedrake_L2_error(u, w), [Control(u), Control(w)])
      G = fem_operator(F)

    loss = 0
    for i in range(len(y_pred)):
        loss += G(y_pred[i], yb[i])

    avg_loss = loss/len(y_pred)

    return avg_loss
