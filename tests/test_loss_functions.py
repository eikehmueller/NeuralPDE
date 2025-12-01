"""Pytest suite to test that the loss functions are working. The tests are

1: the loss between two identical tensors is zero
2: for different scales of functions, the multivariate loss function takes an average
   and thus the loss for random sampling is between 0.5 and 1.
"""

from neural_pde.loss_functions import normalised_mse, normalised_rmse, multivariate_normalised_rmse, normalised_absolute_error
import torch
import numpy as np

def test_lossfns_zero():
    test_tensor = torch.rand((16, 3, 7))
    mse_loss = normalised_mse(test_tensor, test_tensor)
    rmse_loss = normalised_rmse(test_tensor, test_tensor)
    mrmse_loss = multivariate_normalised_rmse(test_tensor, test_tensor)
    ae_loss = normalised_absolute_error(test_tensor, test_tensor)
    assert np.isclose(mse_loss, 0)
    assert np.isclose(rmse_loss, 0)
    assert np.isclose(mrmse_loss, 0)
    assert np.isclose(ae_loss, 0)

def test_multivariate():
    batchsize = 64
    nfunc = 3
    ndofs = 123

    test_tensor1 = torch.rand((batchsize, nfunc, ndofs))
    test_tensor2 = torch.rand((batchsize, nfunc, ndofs))

    test_tensor1[:, 0, :] = 1e3*test_tensor1[:, 0, :]
    test_tensor1[:, 1, :] = 1e-3*test_tensor1[:, 1, :]
    test_tensor2[:, 0, :] = 1e3*test_tensor2[:, 0, :]
    test_tensor2[:, 1, :] = 1e-3*test_tensor2[:, 1, :]


    loss = multivariate_normalised_rmse(test_tensor1, test_tensor2)

    assert 0.5 < loss
    assert loss < 1 
    

