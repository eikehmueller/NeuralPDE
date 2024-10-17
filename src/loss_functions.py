## Module containing our loss functions. For now it is just the normalised L2 error
import torch

def normalised_L2_error(y_pred, yb):
    '''Calculate the normalised L2 error between two pytorch tensors
    :arg y_pred: tensor of size (batchsize, n_func, n_dof)
    :arg yb: tensor of size (batchsize, n_func, n_dof)'''
    loss = torch.mean(
        torch.sum(torch.sum((y_pred - yb)**2, dim=(1, 2))) 
        / torch.sum(torch.sum((yb)**2, dim=(1, 2)))
        )
    return loss