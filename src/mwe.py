"""MWE to illustrate issue that occurs when wrapping Firedrake interpolation operations 
to a set of points"""

import torch
from firedrake import *
from firedrake.__future__ import interpolate
from firedrake.adjoint import continue_annotation, pause_annotation
from pyadjoint import ReducedFunctional, Control
from firedrake.ml.pytorch import fem_operator
from pyadjoint.tape import set_working_tape

# Construct mesh and point mesh onto which we want to interpolate
mesh = UnitSquareMesh(4, 4)
points = [[0.6, 0.1], [0.5, 0.4], [0.7, 0.9]]
vom = VertexOnlyMesh(mesh, points)

# Function spaces on these meshes
fs = FunctionSpace(mesh, "CG", 1)
vertex_only_fs = FunctionSpace(vom, "DG", 0)

# Sizes of function spaces
n_in = int(fs.dof_count)
n_out = int(vertex_only_fs.dof_count)


class Encoder(torch.nn.Module):
    """Differentiable encoder which maps a function from a function space to the points"""

    def __init__(self, fs, vertex_only_fs):
        """Initialise new instance

        :arg fs: original function space
        :arg vertex_only_fs: vertex-only function space to which we want to interpolate
        """
        super().__init__()
        self.in_features = int(fs.dof_count)
        self.out_features = int(vertex_only_fs.dof_count)
        continue_annotation()
        with set_working_tape() as _:
            u = Function(fs)
            interpolator = interpolate(TestFunction(fs), vertex_only_fs)
            self._function_to_patch = fem_operator(
                ReducedFunctional(assemble(action(interpolator, u)), Control(u))
            )
        pause_annotation()

    def forward(self, x):
        """Forward pass.

        The input will be of shape (batch_size, n_in) and the output of size (batch_size, n_out)
        where n_in and n_out are the dimensions of the function spaces.

        :arg x: input tensor
        """
        # apply differentiable interpolation to each tensor in the batch
        return torch.stack([self._function_to_patch(y) for y in torch.unbind(x)])


class Decoder(torch.nn.Module):
    """Differentiable decoder which maps a function from the points to a function space"""

    def __init__(self, fs, vertex_only_fs):
        """Initialise new instance

        :arg fs: original function space
        :arg vertex_only_fs: vertex-only function space to which we want to interpolate
        """
        super().__init__()
        self.in_features = int(vertex_only_fs.dof_count)
        self.out_features = int(fs.dof_count)
        continue_annotation()
        with set_working_tape() as _:
            w = Cofunction(vertex_only_fs.dual())
            interpolator = interpolate(TestFunction(fs), vertex_only_fs)
            self._patch_to_function = fem_operator(
                ReducedFunctional(
                    assemble(action(adjoint(interpolator), w)), Control(w)
                )
            )
        pause_annotation()

    def forward(self, x):
        """Forward pass.

        The input will be of shape (batch_size, n_in) and the output of size (batch_size, n_out)
        where n_in and n_out are the dimensions of the function spaces.

        :arg x: input tensor
        """
        # apply differentiable interpolation to each tensor in the batch
        return torch.stack([self._patch_to_function(y) for y in torch.unbind(x)])


# PyTorch model: linear layer + encoder layer

model = torch.nn.Sequential(
    torch.nn.Linear(in_features=n_in, out_features=n_in).double(),
    Encoder(fs, vertex_only_fs).double(),
    Decoder(fs, vertex_only_fs).double(),
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
    torch.nn.Linear(in_features=n_in, out_features=n_in, bias=False).double(),
    Encoder(fs, vertex_only_fs).double(),
    Decoder(fs, vertex_only_fs).double(),
):
    n_in = layer.in_features
    n_out = layer.out_features
    # extract matrix
    A = np.zeros((n_out, n_in))
    for j in range(n_in):
        x = torch.zeros((1, n_in), dtype=torch.float64)
        x[0, j] = 1.0
        y = layer(x)
        A[:, j] = np.asarray(y.detach())
    x = torch.zeros(n_in, dtype=torch.float64)
    # extract Jacobian
    J = np.asarray(torch.autograd.functional.jacobian(layer, x))
    print("layer = ", str(layer))
    print("||A|| :")
    print(np.linalg.norm(A))
    print()
    print("||J|| :")
    print(np.linalg.norm(J))
    print()
    print("difference ||A-J|| :")
    print(np.linalg.norm(A - J))
    print()
