"""Loss functions; for now it is just the normalised L2 error"""

import torch

__all__ = ["normalised_mse", "normalised_rmse", "multivariate_normalised_rmse", "normalised_absolute_error"]

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
            torch.sum((y_pred - y_target) ** 2,  dim=2)
            / torch.sum((y_target) ** 2, dim=2)
        ), dim=0)

    return torch.mean(loss)

def multivariate_normalised_rmse_with_data(y_pred, y_target, mean, std):
    """Calculate the normalised L2 error between two pytorch tensors

    Compute the batch-average

        1/b sum_{b} rho_b

    where

        rho_b = sqrt(sum_{j,k} (y_pred_{b,j,k} - y_target_{b,j,k})^2 /
                     sum_{j,k} y_target_{b,j,k}^2)

    is the normalised error on each sample. Before averaging, the data is 
    normalised with respect to it's mean and standard deviation

    y_pred = (y_pred - mean) / std

    :arg y_pred: prediction, tensor of size (batchsize, n_func, n_dof)
    :arg y_target: target tensor of size (batchsize, n_func, n_dof)
    :arg mean: mean of the samples, tensor of shape (n_func)
    :arg std: std of the samples, tensor or shape (n_func)
    """

    std  = torch.unsqueeze(std, dim=-1)
    mean  = torch.unsqueeze(mean, dim=-1)
    #mean = torch.unsqueeze(torch.mean(y_target, dim=(0,2)), dim=-1)
    #std = torch.unsqueeze(torch.std(y_target, dim=(0,2)), dim=-1)

    yp = (y_pred - mean) / std
    yt = (y_target - mean) / std

    loss = torch.mean(
        torch.sqrt(
            torch.sum(((yp - yt) ) ** 2,  dim=2) 
            / torch.sum((yt) ** 2, dim=2) 
        ), dim=0)
    

    return torch.mean(loss)

def individual_function_rmse(y_pred, y_target, mean, std):
    """Calculate the function-wise normalised L2 error between two pytorch tensors

    Compute the batch-average

        1/b sum_{b} rho_b

    where

        rho_b = sqrt(sum_{j,k} (y_pred_{b,j,k} - y_target_{b,j,k})^2 /
                     sum_{j,k} y_target_{b,j,k}^2)

    is the normalised error on each sample.

    :arg y_pred: prediction, tensor of size (batchsize, n_func, n_dof)
    :arg y_target: target tensor of size (batchsize, n_func, n_dof)
    :arg mean: mean of the samples, tensor of shape (n_func)
    :arg std: std of the samples, tensor or shape (n_func)
    """

    std  = torch.unsqueeze(std, dim=-1)
    mean  = torch.unsqueeze(mean, dim=-1)

    #mean = torch.unsqueeze(torch.mean(y_target, dim=(0,2)), dim=-1)
    #std = torch.unsqueeze(torch.std(y_target, dim=(0,2)), dim=-1)

    yp = (y_pred - mean) / std
    yt = (y_target - mean) / std

    #print(f"Mean of yp is {torch.mean(yp, dim=(0,2))}")
    #print(f"Std of yp is {torch.std(yp, dim=(0,2))}")
    #print(f"Mean of yt is {torch.mean(yt, dim=(0,2))}")
    #print(f"Std of yt is {torch.std(yt, dim=(0,2))}")


    loss = torch.mean(
        torch.sqrt(
            torch.sum((yp - yt) ** 2,  dim=2)
            / torch.sum((yt) ** 2, dim=2)
        ),  dim=0)

    return loss


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
