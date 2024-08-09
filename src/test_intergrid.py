"""MWE to illustrate issue that occurs when wrapping Firedrake interpolation operations 
to a set of points"""

import pytest
from firedrake import *
import torch
from intergrid import Encoder, Decoder


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
    ### Test 2 - comparing to interpolated meshes ###
    fs, fs_vom = function_spaces
    mesh = fs.mesh()
    u = Function(fs)
    x, y = SpatialCoordinate(mesh)
    u.interpolate(1 + sin(x * pi * 2) * sin(y * pi * 2))
    v = Function(fs_vom)
    v.interpolate(u)

    # dof vectors for u and v
    u_dofs = u.dat.data_ro
    v_dofs = v.dat.data_ro
    u_dofs_tensor = torch.tensor(u_dofs)
    encoder = Encoder(fs, fs_vom).double()
    w_dofs = np.asarray(encoder(u_dofs_tensor))
    assert np.all(np.isclose(v_dofs, w_dofs))
