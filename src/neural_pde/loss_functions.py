"""Loss functions; for now it is just the normalised L2 error"""

import torch

__all__ = ["normalised_mse", "normalised_rmse"]


def normalised_mse(y_pred, y_target):
    """Calculate the normalised L2 squared error between two pytorch tensors

    Compute the batch-average

        1/b sum_{b} rho_b

    where

        rho_b = sum_{j,k} (y_pred_{b,j,k} - y_target_{b,j,k})^2 /
                sum_{j,k} y_target_{b,j,k}^2

    is the normalised error on each sample.

    :arg y_pred: prediction, tensor of size (batchsize, n_func, n_dof)
    :arg y_target: target tensor of size (batchsize, n_func, n_dof)
    """

    loss = torch.mean(
        torch.sum((y_pred - y_target) ** 2, dim=(1, 2))
        / torch.sum((y_target) ** 2, dim=(1, 2))
    )
    return loss


def normalised_rmse(y_pred, y_target):
    """Calculate the normalised L2 error between two pytorch tensors

    Compute the batch-average

        1/b sum_{b} rho_b

    where

        rho_b = sqrt(sum_{j,k} (y_pred_{b,j,k} - y_target_{b,j,k})^2 /
                     sum_{j,k} y_target_{b,j,k}^2)

    is the normalised error on each sample.

    :arg y_pred: prediction, tensor of size (batchsize, n_func, n_dof)
    :arg y_target: target tensor of size (batchsize, n_func, n_dof)
    """

    loss = torch.mean(
        torch.sqrt(
            torch.sum((y_pred - y_target) ** 2, dim=(1, 2))
            / torch.sum((y_target) ** 2, dim=(1, 2))
        )
    )
    return loss
