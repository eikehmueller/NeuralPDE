"""Pytest suite to test that the intergrid functions are working. The tests are

1: check that the Jacobian of the encoder and decoder are correct
2: check that the shape of the Jacobian is as expected
3: check that the jacobian of the decoder is the adjoint of the jacobian of the encoder
4: check that the encoder gives the same results as manual interpolation of the functions
5: check that interpolation using Encoder is the same as manual interpolation
    for an input tensor with the shape (n, n_in). 
6: check that interpolation using Encoder is the same as manual interpolation
    for an input tensor with the shape (batch_size, n, n_in). 

"""

import pytest
from firedrake import *
import torch
from neural_pde.intergrid import Encoder, Decoder
from firedrake.ml.pytorch import to_torch


@pytest.fixture
def function_spaces():
    """Construct function spaces for tests"""

    mesh = UnitSquareMesh(4, 4)
    points = [[0.6, 0.1], [0.5, 0.4], [0.7, 0.9]]
    vom = VertexOnlyMesh(mesh, points)

    # Function spaces on these meshes
    fs = FunctionSpace(mesh, "CG", 1)
    fs_vom = FunctionSpace(vom, "DG", 0)
    return fs, fs_vom


@pytest.mark.parametrize("Operator", [Encoder, Decoder])
def test_jacobian(function_spaces, Operator):
    """Check that the Jacobian of the encoder and decoder are correct"""
    # Sizes of function spaces
    fs, fs_vom = function_spaces
    operator = Operator(fs, fs_vom).double()
    n_in = operator.in_features
    n_out = operator.out_features
    # extract matrix
    A = np.zeros((n_out, n_in))
    for j in range(n_in):
        x = torch.zeros(n_in, dtype=torch.float64)
        x[j] = 1.0
        y = operator(x)
        A[:, j] = np.asarray(y.detach())
    x = torch.zeros(n_in, dtype=torch.float64)
    # extract Jacobian
    J = np.asarray(torch.autograd.functional.jacobian(operator, x))
    assert (
        np.isclose(np.linalg.norm(A - J), 0.0, rtol=1e-12, atol=1e-12)
        and np.linalg.norm(A) > 1
    )


@pytest.mark.parametrize("Operator", [Encoder, Decoder])
def test_jacobian_shape(function_spaces, Operator):
    """Check that the shape of the Jacobian is as expected"""
    fs, fs_vom = function_spaces
    operator = Operator(fs, fs_vom).double()
    n_in = operator.in_features
    n_out = operator.out_features
    x = torch.zeros(n_in, dtype=torch.float64)
    J = np.asarray(torch.autograd.functional.jacobian(operator, x))
    assert J.shape == (n_out, n_in)


def test_jacobians_are_adjoint(function_spaces):
    """Check that the jacobian of the decoder is the adjoint of the jacobian
    of the encoder"""
    fs, fs_vom = function_spaces
    encoder = Encoder(fs, fs_vom).double()
    decoder = Decoder(fs, fs_vom).double()
    n_in = encoder.in_features
    n_out = decoder.in_features
    x = torch.zeros(n_in, dtype=torch.float64)
    y = torch.zeros(n_out, dtype=torch.float64)
    # Jacobian of encoder
    J_encoder = np.asarray(torch.autograd.functional.jacobian(encoder, x))
    # Jacobian of decoder
    J_decoder = np.asarray(torch.autograd.functional.jacobian(decoder, y))
    assert np.all(np.isclose(J_encoder.T, J_decoder))


def test_manual_interpolation(function_spaces):
    """Check that the encoder gives the same results as manual interpolation
    of the functions"""
    fs, fs_vom = function_spaces
    mesh = fs.mesh()
    u = Function(fs)
    x, y = SpatialCoordinate(mesh)
    u.interpolate(1 + sin(x * pi * 2) * sin(y * pi * 2))
    v = Function(fs_vom)
    v.interpolate(u)

    # dof vectors for u and v
    u_dofs = to_torch(u).flatten()
    v_dofs = to_torch(v).flatten()
    encoder = Encoder(fs, fs_vom).double()
    w_dofs = np.asarray(encoder(u_dofs))
    assert np.all(np.isclose(v_dofs, w_dofs))


def test_manual_interpolation_4d(function_spaces):
    """Check that the encoder gives the same results as manual interpolation
    of the functions in multiple dimensions"""
    fs, fs_vom = function_spaces
    u = Function(fs)
    v = Function(fs_vom)
    rng = np.random.default_rng(seed=3426197)
    n1 = 3
    n2 = 4
    n3 = 5
    n_in = fs.dof_count
    n_out = fs_vom.dof_count
    u_dofs = np.empty(shape=(n1, n2, n3, n_in))
    v_dofs = np.empty(shape=(n1, n2, n3, n_out))
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                with u.dat.vec as u_vec:
                    u_vec[:] = rng.normal(size=n_in)
                    u_dofs[i1, i2, i3, :] = u_vec[:]
                v.interpolate(u)
                with v.dat.vec_ro as v_vec:
                    v_dofs[i1, i2, i3, :] = v_vec[:]
    encoder = Encoder(fs, fs_vom).double()
    w_dofs = np.asarray(encoder(torch.tensor(u_dofs)))
    assert np.all(np.isclose(v_dofs, w_dofs))


def test_two_dimensions(function_spaces):
    """Check that interpolation using Encoder is the same as manual interpolation
    for an input tensor with the shape (n, n_in). Output should be of the shape
    (n, n_vom)"""
    fs, fs_vom = function_spaces
    encoder = Encoder(fs, fs_vom).double()
    mesh = fs.mesh()
    u1 = Function(fs)
    u2 = Function(fs)
    x, y = SpatialCoordinate(mesh)
    u1.interpolate(1 + sin(x * pi * 2) * sin(y * pi * 2))
    u2.interpolate(2 + sin(x * pi * 2) * cos(y * pi * 2))
    u_dofs_tensor = torch.vstack((to_torch(u1).flatten(), to_torch(u2).flatten()))
    v1 = Function(fs_vom)
    v1.interpolate(u1)
    v2 = Function(fs_vom)
    v2.interpolate(u2)
    v_dofs_tensor = torch.vstack((to_torch(v1).flatten(), to_torch(v2).flatten()))
    w_dofs_tensor = np.asarray(encoder(u_dofs_tensor))
    assert np.all(np.isclose(v_dofs_tensor, w_dofs_tensor))


def test_three_dimensions(function_spaces):
    """Check that interpolation using Encoder is the same as manual interpolation
    for an input tensor with the shape (batch_size, n, n_in). Output should be of the shape
    (batch_size, n, n_vom)"""
    fs, fs_vom = function_spaces
    encoder = Encoder(fs, fs_vom).double()
    mesh = fs.mesh()
    u1 = Function(fs)
    u2 = Function(fs)
    x, y = SpatialCoordinate(mesh)
    u1.interpolate(1 + sin(x * pi * 2) * sin(y * pi * 2))
    u2.interpolate(2 + sin(x * pi * 2) * cos(y * pi * 2))
    u_dofs_tensor = torch.vstack((to_torch(u1).flatten(), to_torch(u2).flatten()))
    u_dofs_batches = torch.vstack((u_dofs_tensor, u_dofs_tensor, u_dofs_tensor))
    v1 = Function(fs_vom)
    v1.interpolate(u1)
    v2 = Function(fs_vom)
    v2.interpolate(u2)
    v_dofs_tensor = torch.vstack((to_torch(v1).flatten(), to_torch(v2).flatten()))
    v_dofs_batches = torch.vstack((v_dofs_tensor, v_dofs_tensor, v_dofs_tensor))
    w_dofs_batches = np.asarray(encoder(u_dofs_batches))
    assert np.all(np.isclose(v_dofs_batches, w_dofs_batches))
