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

def multivariate_normalised_rmse(y_pred, y_target):
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
            torch.sum((y_pred - y_target) ** 2, dim=2)
            / torch.sum((y_target) ** 2, dim=2)
        ), dim=0
    )
    return torch.sum(loss)

def normalised_absolute_error(y_pred, y_target):
    """Calculate the normalised L2 error between two pytorch tensors

    Compute the batch-average

        1/b sum_{b} rho_b

    where

        rho_b = sum_{j,k} |y_pred_{b,j,k} - y_target_{b,j,k}| /
                     sum_{j,k} |y_target_{b,j,k}|

    is the normalised error on each sample.

    :arg y_pred: prediction, tensor of size (batchsize, n_func, n_dof)
    :arg y_target: target tensor of size (batchsize, n_func, n_dof)
    """

    loss = torch.mean(
            torch.sum(torch.abs(y_pred - y_target) , dim=(1, 2))
            / torch.sum(torch.abs(y_target), dim=(1, 2))
    )
    return loss

'''
test_tensor = torch.rand((16, 3, 7))
summed = torch.sum(test_tensor**2, dim=2)
print(summed.shape)
mean = torch.mean(summed, dim=0)
print(mean)
print(sum(mean))
'''