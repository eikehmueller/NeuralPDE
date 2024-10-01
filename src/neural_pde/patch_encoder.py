"""Patch encoder. Encodes the field information into the latent space"""

from firedrake import *
from firedrake.adjoint import *
import torch
from neural_pde.intergrid import Encoder


class PatchEncoder(torch.nn.Module):
    """
    This class inherits from class torch.nn.Module

    Collective encoding to latent space

    The input X has shape

        (B,n_{func},n_{dof}) or (n_{func},n_{dof})

    where the batch-dimension is of size B, n_{func} = n_{dynamic} + n_{ancillary} is the total
    number of functions, consisting of both dynamic and ancillary functions. n_{dof} is the
    number of unknowns per function, as defined by the function space.

    The output has shape

        (B,n_{patches},d_{lat}) or (n_{patches}, d_{lat})

    where d_{lat} = d_{lat}^{dynamic} + d_{lat}^{ancillary} is the dimension of the latent
    space.

    To achieve this, each of the input functions is first projected to a patch covering to obtain
    a tensor of shape

        (B,n_{patches},n_{func},n_{dof per patch})

    where n_{patches} and n_{dof per patch} depend on the SphericalPatchCovering. This is then
    mapped to latent space with a learnable embedding.

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

        mesh = fs.mesh()  # extract the mesh from the functionspace
        points = spherical_patch_covering.points.reshape(
            (spherical_patch_covering.n_points, 3)
        )  # extract the points from the spherical_patch_covering
        self._npatches = spherical_patch_covering.n_patches  # number of patches
        self._patchsize = (
            spherical_patch_covering.patch_size
        )  # number of points per patch
        vertex_only_mesh = VertexOnlyMesh(mesh, points)
        vertex_only_fs = FunctionSpace(
            vertex_only_mesh, "DG", 0
        )  # we can only use DG0 for a vom.

        continue_annotation()
        with set_working_tape() as _:
            self._function_to_patch = Encoder(fs, vertex_only_fs)
        pause_annotation()

        self._n_dynamic = n_dynamic

    def forward(self, x):
        """Forward map

        :arg x: input
        """
        # Part I: interpolation to VOM
        # input is a (n_func, ndof) tensor

        if x.dim() == 2:
            x = torch.stack(
                [
                    torch.reshape(
                        self._function_to_patch.forward(z),
                        (self._npatches, self._patchsize),
                    )
                    for z in torch.unbind(x)
                ]
            )

            # permute axes to obtain tensor or shape (n_{patches},n_{func},n_{dof per patch})
            x = torch.permute(x, (1, 0, 2))

            # Part II: encoding on patches
            x_ancillary = self._ancillary_encoder_model(x[..., self._n_dynamic :, :])
            # start slicing from index n_dynamic. Assumes that n_func is of shape [dyn, anc, anc anc]
            x_dynamic = self._dynamic_encoder_model(x)
            x = torch.cat((x_dynamic, x_ancillary), dim=-1)

            # x has shape (npatch, dim (ancilliary + dynamic))
            return x
        else:
            x = torch.stack(
                [
                    torch.stack(
                        [
                            torch.reshape(  # input, shape
                                self._function_to_patch.forward(z),
                                (self._npatches, self._patchsize),
                            )
                            for z in torch.unbind(y)
                        ]
                    )
                    for y in torch.unbind(x)
                ]
            )

            # permute axes to obtain tensor or shape (B,n_{patches},n_{func},n_{dof per patch})
            x = torch.permute(x, (0, 2, 1, 3))

            # Part II: encoding on patches
            x_ancillary = self._ancillary_encoder_model(x[..., self._n_dynamic :, :])
            # start slicing from index n_dynamic. Assumes that n_func is of shape [dyn, anc, anc anc]
            x_dynamic = self._dynamic_encoder_model(x)
            x = torch.cat((x_dynamic, x_ancillary), dim=-1)

            # x has shape (B, npatch, dim (ancilliary + dynamic))
            return x
