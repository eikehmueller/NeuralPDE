import torch
from firedrake import *
from firedrake.adjoint import *
from firedrake.ml.pytorch import fem_operator
continue_annotation()


mesh = RectangleMesh(10, 10, 1, 1) # create the mesh
V = FunctionSpace(mesh, "CG", 1) # define the function space

#### defining the L2 error for the firedrake function ###
def firedrake_L2_error(u, w):
   return assemble((u - w) ** 2 * dx)**0.5

u = Function(V)
w = Function(V)
with set_working_tape() as _:
   F = ReducedFunctional(firedrake_L2_error(u, w), [Control(u), Control(w)])
   G = fem_operator(F)


batchsize=16
dof_len=len(u.dat.data[:])

#### an example batch in a training loop #####
X = torch.rand(dof_len).double()
Y = torch.rand(dof_len).double()
        
loss = G(X, Y)
print(loss)