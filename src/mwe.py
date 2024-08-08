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

    @staticmethod
    def function_to_patch(metadata, x):
        u = from_torch(x, V=metadata["fs"])
        w = assemble(action(metadata["interpolator"], u))
        return to_torch(w)

    @staticmethod
    def patch_to_function(metadata, x):
        w = from_torch(x, V=metadata["vertex_only_fs"].dual())
        u = assemble(action(adjoint(metadata["interpolator"]), w))
        return to_torch(u)

    @staticmethod
    def forward(ctx, metadata, *xP):
        ctx.metadata.update(metadata)
        if metadata["reverse"]:
            return InterpolatorWrapper.patch_to_function(metadata, xP[0])
        else:
            return InterpolatorWrapper.function_to_patch(metadata, xP[0])

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.metadata["reverse"]:
            return None, InterpolatorWrapper.function_to_patch(
                ctx.metadata, grad_output
            )
        else:
            return None, InterpolatorWrapper.patch_to_function(
                ctx.metadata, grad_output
            )


def interpolator_wrapper(fs, vertex_only_fs, reverse=False):
    interpolator = interpolate(TestFunction(fs), vertex_only_fs)
    metadata = {
        "fs": fs,
        "vertex_only_fs": vertex_only_fs,
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
        return torch.stack([self._patch_to_function(y) for y in torch.unbind(x)])


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
    x.unsqueeze(dim=0)
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
