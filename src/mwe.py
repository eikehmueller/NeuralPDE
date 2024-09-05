"""MWE to illustrate issue that occurs when wrapping Firedrake interpolation operations 
to a set of points"""

from firedrake import *
import torch


#from intergrid import Encoder, Decoder

#for i in range(torch.cuda.device_count()):
#   print(torch.cuda.get_device_properties(i).name)

#print(torch.cuda.device_count())
'''
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
'''