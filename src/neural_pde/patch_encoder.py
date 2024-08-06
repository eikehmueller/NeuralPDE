"""Patch encoder. Encodes the field information into the latent space"""

from firedrake import *
from firedrake.ml.pytorch import fem_operator
from pyadjoint import ReducedFunctional, Control
from firedrake.adjoint import *
from firedrake.__future__ import interpolate
import torch


class PatchEncoder(torch.nn.Module):
    """Collective encoding to latent space

    The input X has shape

        (B,n_{func},n_{dof})

    where the batch-dimension is of size B. n_{func} = n_{dynamic} + n_{ancillary} is the total
    number of functions, consisting of both dynamic and ancillary functions. n_{dof} is the
    number of unknowns per function, as defined by the function space.

    The output has shape

        (B,n_{patches},d_{lat})

    where d_{lat} = d_{lat}^{dynamic} + d_{lat}^{ancillary} is the dimension of the latent
    space.

    To achieve this, each of the input function is first projected to a patch covering to obtain
    a tensor of shape

        (B,n_{patches},n_{func},n_{dof per patch})

    where n_{patches} and n_{dof per patch} depend on the SphericalPatchCovering. This is then mappend to latent space with a learnable embedding.

    This embedding has block-structure in the sense that

        [b,i,:,:] gets mapped to [b,i,:d_{latent}]

    by applying dynamic_encoder_model for each index pair (b,i)

    and

        [b,i,n_{dynamic}:,:] gets mapped to [p,i,d_{latent}:].

    by applying the ancillary_encoder_model for each index pair (b,i).

    This means that the latent variables in the processor will depend both on the dynamic- and
    on the ancillary fields on the patches whereas the ancillary variables in the processor
    will only depend on the ancillary fields on the patches.
    """

    def __init__(
        self,
        fs,
        spherical_patch_covering,
        dynamic_encoder_model,
        ancillary_encoder_model,
        n_dynamic,
    ):
        """Initialise instance

        :arg fs: function space
        :arg spherical_patch_covering: the patch covering for the intermediate
            stage
        :arg dynamic_encoder_model: model that maps tensors of shape
            (n_{dynamic}+n_{ancillary},patch_size) to tensors of shape (d_{latent},)
        :arg ancillary_encoder_model: model that maps tensors of shape
            (n_{ancillary},patch_size) to tensors of shape (d_{ancillary},)
        :arg n_dynamic: number of dynamic functions
        """
        super().__init__()
        self._dynamic_encoder_model = dynamic_encoder_model
        self._ancillary_encoder_model = ancillary_encoder_model

        mesh = fs.mesh()
        points = spherical_patch_covering.points.reshape(
            (spherical_patch_covering.n_points, 3)
        )
        self._npatches = spherical_patch_covering.n_patches
        self._patchsize = spherical_patch_covering.patch_size
        vertex_only_mesh = VertexOnlyMesh(mesh, points)
        vertex_only_fs = FunctionSpace(vertex_only_mesh, "DG", 0)

        continue_annotation()
        with set_working_tape() as _:
            u = Function(fs)
            interpolator = interpolate(TestFunction(fs), vertex_only_fs)
            self._function_to_patch = fem_operator(
                ReducedFunctional(assemble(action(interpolator, u)), Control(u))
            )
        pause_annotation()
        self._n_dynamic = n_dynamic

    def forward(self, x):
        """Forward map

        :arg x: input
        """
        # Part I: interpolation to VOM
        x = torch.stack(
            [
                torch.stack(
                    [
                        torch.reshape(
                            self._function_to_patch(z),
                            (self._npatches, self._patchsize),
                        )
                        for z in torch.unbind(y)
                    ]
                )
                for y in torch.unbind(x)
            ]
        )
        # (B,n_{func},n_{dof})
        # permute axes to obtain tensor or shape (B,n_{patches},n_{func},n_{dof per patch})
        x = torch.permute(x, (0, 2, 1, 3))
        # Part II: encoding on patches

        x_ancillary = self._ancillary_encoder_model(x[..., self._n_dynamic :, :])
        x_dynamic = self._dynamic_encoder_model(x)
        x = torch.cat((x_dynamic, x_ancillary), dim=-1)
        return x
