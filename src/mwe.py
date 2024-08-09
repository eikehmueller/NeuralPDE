"""MWE to illustrate issue that occurs when wrapping Firedrake interpolation operations 
to a set of points"""

import torch
from firedrake import *
from firedrake.__future__ import interpolate
from firedrake.ml.pytorch import from_torch, to_torch


class InterpolatorWrapper(torch.autograd.Function):
    """Differentiable PyTorch wrapper around Firedrake interpolation

    Allows interpolation of functions between two Firedrake function
    spaces. Both the forward and adjoint operation are implemented. Since
    interpolation is a linear operation y = A.x, the gradient is propagated
    like grad_in = A^T.grad_out
    """

    @staticmethod
    def interpolate(metadata, x):
        """Interpolate a function given by its dof-vector

        :arg metadata: information on function spaces
        :arg x: dof-vector of function on input function space
        """
        u = from_torch(x, V=metadata["fs_from"])
        w = assemble(action(metadata["interpolator"], u))
        return torch.flatten(to_torch(w))

    @staticmethod
    def adjoint_interpolate(metadata, x):
        """Map a dual vector on the output function space to the input space

        :arg metadata: information on function spaces
        :arg x: dof-vector of dual function on output function space
        """
        w = from_torch(x, V=metadata["fs_to"].dual())
        u = assemble(action(adjoint(metadata["interpolator"]), w))
        return torch.flatten(to_torch(u))

    @staticmethod
    def forward(ctx, metadata, x):
        """Forward map

        Computes y = A.x if x is a dof-vector on the input function space
        or y = A^T.x if x is a dual vector on the output function space

        :arg ctx: context
        :arg metadata: information on function spaces
        :arg x: tensor to operate on
        """
        ctx.metadata.update(metadata)
        if metadata["reverse"]:
            return InterpolatorWrapper.adjoint_interpolate(metadata, x)
        else:
            return InterpolatorWrapper.interpolate(metadata, x)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward map

        :arg ctx: context
        :arg grad_output: gradient with respect out output variables
        """
        if ctx.metadata["reverse"]:
            return None, InterpolatorWrapper.interpolate(ctx.metadata, grad_output)
        else:
            return None, InterpolatorWrapper.adjoint_interpolate(
                ctx.metadata, grad_output
            )


def interpolator_wrapper(fs_from, fs_to, reverse=False):
    """Construct interpolation object from

    Depending on the value of reverse, this either interpolates the
    dof-vectors of functions from fs_from to fs_to or the dof-vectors of
    dual functions from fs_to to fs_from

    :arg fs_from: function space which we want to interpolate from
    :arg fs_to: function space which we want to interpolate to
    """
    interpolator = interpolate(TestFunction(fs_from), fs_to)
    metadata = {
        "fs_from": fs_from,
        "fs_to": fs_to,
        "interpolator": interpolator,
        "reverse": reverse,
    }
    return partial(InterpolatorWrapper.apply, metadata)


class Encoder(torch.nn.Module):
    """Differentiable encoder which interpolates to a different function space

    This maps dof-vectors of functions u from one function space to
    dof-vectors of another function v on another function space in such a way
    that the original function is interpolated to the target space, i.e.
    v = interpolate(u).
    """

    def __init__(self, fs_from, fs_to):
        """Initialise new instance

        :arg fs_from: original function space
        :arg fs_to: target function space that we want to interpolate to
        """
        super().__init__()
        self.in_features = int(fs_from.dof_count)
        self.out_features = int(fs_to.dof_count)
        self._interpolate = interpolator_wrapper(fs_from, fs_to, reverse=False)

    def forward(self, x):
        """Forward pass

        The input will be of shape (batch_size, n_in) and the output of size (batch_size, n_out)
        where n_in and n_out are the dimensions of the function spaces.

        :arg x: input tensor
        """
        return self._forward(x)

    def _forward(self, x):
        """Recursively apply forward pass

        Unbind the input, recursively call the function and stack the outputs.
        If x has dimension 1, apply the interpolation.
        """
        if x.dim() == 1:
            return self._interpolate(x)
        else:
            return torch.stack([self._forward(y) for y in torch.unbind(x)])


class Decoder(torch.nn.Module):
    """Differentiable encoder which implements the adjoint interpolation to a function space

    Maps the dof-vectors of a dual function v* on the target function space
    to the dof-vectors of a dual function u* on the original function space
    in such a way that v = interpolate(u). Hence, this class realises the
    adjoint of the Encoder operation.
    """

    def __init__(self, fs_from, fs_to):
        """Initialise new instance

        :arg fs_from: original function space that we want to interpolate from
        :arg fs_to: target function space to which we want to interpolate
        """
        super().__init__()
        self.in_features = int(fs_to.dof_count)
        self.out_features = int(fs_from.dof_count)
        self._adjoint_interpolate = interpolator_wrapper(fs_from, fs_to, reverse=True)

    def forward(self, x):
        """Forward pass.

        The input will be of shape (batch_size, n_in) and the output of size (batch_size, n_out)
        where n_in and n_out are the dimensions of the function spaces.

        :arg x: input tensor
        """
        return self._forward(x)

    def _forward(self, x):
        """Recursively apply forward pass

        Unbind the input, recursively call the function and stack the outputs.
        If x has dimension 1, apply the revserve interpolation.
        """
        if x.dim() == 1:
            return self._adjoint_interpolate(x)
        else:
            return torch.stack([self._forward(y) for y in torch.unbind(x)])


# Construct meshes onto which we want to interpolate
mesh = UnitSquareMesh(3, 3)

points = [[0.6, 0.1], [0.5, 0.4], [0.7, 0.9]]
vom = VertexOnlyMesh(mesh, points)

# Function spaces on these meshes
fs = FunctionSpace(mesh, "CG", 1)
fs_vom = FunctionSpace(vom, "DG", 0)

# Sizes of function spaces
n_in = len(Function(fs).dat.data)
n_out = len(Function(fs_vom).dat.data)

model = torch.nn.Sequential(
    torch.nn.Linear(in_features=n_in, out_features=n_in).double(),
    Encoder(fs, fs_vom).double(),
    Decoder(fs, fs_vom).double(),
)

# Input and target tensors (random)
batch_size = 4
X = torch.tensor(
    np.random.normal(
        size=(
            batch_size,
            n_in,
        ),
    )
)
y_target = torch.tensor(
    np.random.normal(
        size=(
            batch_size,
            n_in,
        ),
    )
)

# PyTorch optimiser and loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

# Do a single gradient computation
optimizer.zero_grad()
y = model(X)
loss = loss_fn(y, y_target)
loss.backward()

### Test 1 - comparing A and J ###
for layer in (
    torch.nn.Linear(in_features=3, out_features=7, bias=False).double(),
    Encoder(fs, fs_vom).double(),
    Decoder(fs, fs_vom).double(),
):
    n_in = layer.in_features
    n_out = layer.out_features

    # extract matrix
    A = np.zeros((n_out, n_in))
    for j in range(n_in):
        x = torch.zeros(n_in, dtype=torch.float64)
        x[j] = 1.0
        y = layer(x)
        A[:, j] = np.asarray(y.detach())
    x = torch.zeros(n_in, dtype=torch.float64)
    # extract Jacobian
    J = np.asarray(torch.autograd.functional.jacobian(layer, x))
    print("layer = ", str(layer))
    print(f"A is {type(A)}")
    print(f"Shape of A is {A.shape}")
    print("||A|| :")
    print(np.linalg.norm(A))
    print()
    print(f"Shape of J is {J.shape}")
    print("||J|| :")
    print(np.linalg.norm(J))
    print()
    print("difference ||A-J|| :")
    print(np.linalg.norm(A - J))
    print()

### Test 2 - comparing to interpolated meshes ###
u = Function(fs)
x, y = SpatialCoordinate(mesh)
u.interpolate(1 + sin(x * pi * 2) * sin(y * pi * 2))
v = Function(fs_vom)
v.interpolate(u)

# dof vectors for u and v
u_dofs = u.dat.data_ro
print(f"u_dofs are {u_dofs}")
v_dofs = v.dat.data_ro
print(f"v_dofs are {v_dofs}")

# use u_dofs as input for encoder
# THE INPUT HAS TO BE OF THE FORM (BATCH, N_IN)
u_dofs_tensor = torch.tensor(u_dofs).unsqueeze(0)
model = Encoder(fs, fs_vom).double()
model_output = model(u_dofs_tensor)
new_v_dofs = model_output.numpy()

print(f"new_v_dofs are {new_v_dofs}")
print(f"difference ||v_dofs - new_v_dofs|| = {np.linalg.norm(v_dofs - new_v_dofs)}")
