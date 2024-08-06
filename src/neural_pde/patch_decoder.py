"""Patch decoder. Decoces information from latent space back to a function on the sphere"""

from firedrake import *
from firedrake.ml.pytorch import fem_operator
from pyadjoint import ReducedFunctional, Control
from firedrake.adjoint import *
from firedrake.__future__ import interpolate
import torch


class PatchDecoder(torch.nn.Module):
    """Collective decoding from latent space

    The input X has shape

        (B,n_{patches},d_{lat})

    where the batch-dimension is of size B. d_{lat} = d_{lat}^{dynamic} + d_{lat}^{ancillary}
    is the dimension of the latent space, consisting of both dynamic and ancillary variables.

    The output has shape

        (B,n_{out},n_{dof})

    where n_{out} is the number of output functions.

    To achieve this, each of the input function is first mapped to a function on
    the patch covering with a learnable embedding.

    This embedding has block-structure in the sense that

        [b,i,:] gets mapped to [b,,:,i,:]

    by applying decoder_model for each sample b and each patch i. This will result in a number
    of functions on the patch covering, which is a tensor of shape

        (B,n_{out},n_{patches},n_{dof per patch})

    where n_{patches} and n_{dof per patch} depend on the SphericalPatchCovering. This tensor
    is then mapped to a tensor of shape (B,n_{out},n_{dof}) by using the adjoint of a
    projection to a VOM.
    """

    def __init__(
        self,
        fs,
        spherical_patch_covering,
        decoder_model,
    ):
        """Initialise instance

        :arg fs: function space
        :arg spherical_patch_covering: the patch covering for the intermediate
            stage
        :arg decoder_model: model that maps tensors of shape
            (d_{lat},) to tensors of shape (n_{out},patch_size)
        """
        super().__init__()
        self._decoder_model = decoder_model
        self._ancillary_encoder_model = decoder_model

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
            w = Cofunction(vertex_only_fs.dual())
            interpolator = interpolate(TestFunction(fs), vertex_only_fs)
            self._patch_to_function = fem_operator(
                ReducedFunctional(
                    assemble(action(adjoint(interpolator), w)), Control(w)
                )
            )
        continue_annotation()

    def forward(self, x):
        """Forward map

        :arg x: input
        """
        # Part I: encoding on patches
        x = self._decoder_model(x)
        # permute axes to obtain tensor or shape (B,n_{out},n_{patches},n_{dof per patch})
        x = torch.permute(x, (0, 2, 1, 3))
        # Part II: (adjoint) interpolation from VOM to spherical function space
        x = torch.stack(
            [
                torch.stack(
                    [
                        torch.flatten(
                            self._patch_to_function(
                                torch.flatten(z, start_dim=-2, end_dim=-1)
                            )
                        )
                        for z in torch.unbind(y)
                    ]
                )
                for y in torch.unbind(x)
            ]
        )
        return x
