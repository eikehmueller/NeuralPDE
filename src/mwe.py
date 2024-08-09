"""MWE to illustrate issue that occurs when wrapping Firedrake interpolation operations 
to a set of points"""

import torch
from firedrake import *
from firedrake.__future__ import interpolate
from firedrake.ml.pytorch import from_torch, to_torch


# Construct meshes onto which we want to interpolate
mesh = UnitSquareMesh(3, 3)

points = [[0.6, 0.1], [0.5, 0.4], [0.7, 0.9]]
vom = VertexOnlyMesh(mesh, points)

# Function spaces on these meshes
fs = FunctionSpace(mesh, "CG", 1)
fs2 = FunctionSpace(vom, "DG", 0)

# Sizes of function spaces
n_in = len(Function(fs).dat.data)
n_out = len(Function(fs2).dat.data)


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
        return to_torch(w)

    @staticmethod
    def adjoint_interpolate(metadata, x):
        """Map a dual vector on the output function space to the input space

        :arg metadata: information on function spaces
        :arg x: dof-vector of dual function on output function space
        """
        w = from_torch(x, V=metadata["fs_to"].dual())
        u = assemble(action(adjoint(metadata["interpolator"]), w))
        return to_torch(u)

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
    """Differentiable encoder which maps a function from a function space to
    another function space"""

    def __init__(self, fs, fs2):
        """Initialise new instance

        :arg fs: original function space
        :arg fs2: vom function space to which we want to interpolate
        """
        super().__init__()
        self.in_features = int(fs.dof_count)
        self.out_features = int(fs2.dof_count)
        self._function_to_patch = interpolator_wrapper(fs, fs2, reverse=False)

    def forward(self, x):
        """Forward pass.

        The input will be of shape (batch_size, n_in) and the output of size (batch_size, n_out)
        where n_in and n_out are the dimensions of the function spaces.

        :arg x: input tensor
        """
        # apply differentiable interpolation to each tensor in the batch
        output_tensor = self._function_to_patch(torch.unbind(x)[0]).squeeze()
        if x.shape[0] == 1:
            pass
        else:
            for i in range(1, x.shape[0]):  # the number of batches we are doing
                batch_data = self._function_to_patch(torch.unbind(x)[i])
                output_tensor = torch.vstack([output_tensor, batch_data])
        return output_tensor


class Decoder(torch.nn.Module):
    """Differentiable decoder which maps a function from the points to a function space"""

    def __init__(self, fs, fs2):
        """Initialise new instance

        :arg fs: original function space
        :arg fs2:function space to which we want to interpolate
        """
        super().__init__()
        self.in_features = int(fs2.dof_count)
        self.out_features = int(fs.dof_count)
        self._patch_to_function = interpolator_wrapper(fs, fs2, reverse=True)

    def forward(self, x):
        """Forward pass.

        The input will be of shape (batch_size, n_in) and the output of size (batch_size, n_out)
        where n_in and n_out are the dimensions of the function spaces.

        :arg x: input tensor
        """
        # apply differentiable interpolation to each tensor in the batch
        output_tensor = self._patch_to_function(torch.unbind(x)[0]).squeeze()
        if x.shape[0] == 1:
            pass
        else:
            for i in range(1, x.shape[0]): # the number of batches we are doing
                batch_data = self._patch_to_function(torch.unbind(x)[i])
                output_tensor = torch.vstack([output_tensor, batch_data])
        return output_tensor


# PyTorch model: linear layer + encoder layer

# model(theta) = f ( g (theta) )
# dmodel/dtheta
model = torch.nn.Sequential(
    torch.nn.Linear(in_features=n_in, out_features=n_in).double(),
    Encoder(fs, fs2).double(),
    Decoder(fs, fs2).double(),
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
    Encoder(fs, fs2).double(),
    Decoder(fs, fs2).double(),
):
    n_in = layer.in_features
    n_out = layer.out_features

    # extract matrix
    A = np.zeros((n_out, n_in))
    for j in range(n_in):
        x = torch.zeros((1, n_in), dtype=torch.float64)
        x[0, j] = 1.0
        x.unsqueeze(dim=0)
        y = layer(x)
        A[:, j] = np.asarray(y.detach())
    x = torch.zeros(n_in, dtype=torch.float64)
    # extract Jacobian
    J = np.asarray(
        torch.autograd.functional.jacobian(layer, x.unsqueeze(dim=0))
    ).squeeze()
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
u.interpolate(1 + sin(x*pi*2)*sin(y*pi*2))
v = Function(fs2)
v.interpolate(u)

# dof vectors for u and v
u_dofs = u.dat.data_ro
print(f'u_dofs are {u_dofs}')
v_dofs = v.dat.data_ro
print(f'v_dofs are {v_dofs}')

# use u_dofs as input for encoder
# THE INPUT HAS TO BE OF THE FORM (BATCH, N_IN)
u_dofs_tensor = torch.tensor(u_dofs).unsqueeze(0)
model = Encoder(fs, fs2).double()
model_output = model(u_dofs_tensor)
new_v_dofs = model_output.numpy()

print(f'new_v_dofs are {new_v_dofs}')
print(f'difference ||v_dofs - new_v_dofs|| = {np.linalg.norm(v_dofs - new_v_dofs)}')
