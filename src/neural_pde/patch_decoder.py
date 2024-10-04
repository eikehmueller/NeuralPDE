"""Patch decoder. Decodes information from latent space back to a function on the sphere"""

from firedrake import *

# from firedrake.ml.pytorch import fem_operator
# from pyadjoint import ReducedFunctional, Control
from firedrake.adjoint import *
import torch
from neural_pde.intergrid import Decoder


class PatchDecoder(torch.nn.Module):
    """Collective decoding from latent space

    The input X has shape

        (B,n_{patches},d_{lat}) or (n_{patches}, d_{lat})

    where the batch-dimension is of size B. d_{lat} = d_{lat}^{dynamic} + d_{lat}^{ancillary}
    is the dimension of the latent space, consisting of both dynamic and ancillary variables.

    The output has shape

        (B,n_{out},n_{dof}) or (n_{out}, n_{dof})

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
            self._patch_to_function = Decoder(fs, vertex_only_fs)
        pause_annotation()

    def forward(self, x):
        """Forward map

        :arg x: input
        """
        device = x.device

        print(f'Input for decoder has size {x.size()}')

        if x.dim() == 2:
            # Part I: encoding on patches
            x = self._decoder_model(x)
            x = torch.permute(x, (1, 0, 2))
            # Part II: (adjoint) interpolation from VOM to spherical function space
            x = torch.stack(
                [
                    torch.flatten(
                        self._patch_to_function.forward(
                            torch.flatten(z, start_dim=-2, end_dim=-1)
                        ).to(device)
                    )
                    for z in torch.unbind(x)
                ]
            )
            return x
        else:
            # Part I: encoding on patches
            #print(f'After applying decoding model, x should have size [32, 20, 1, 6]')
            #print(f'However, 20 and 6 do not multiply to give 42')
            x = self._decoder_model(x)
            print(f'After applying decoding model, x has size {x.size()}')
            x = torch.permute(x, (0, 2, 1, 3))
            # Part II: (adjoint) interpolation from VOM to spherical function space
            print(f'After permutation, x has size {x.size()}')
            print(f'After permutation, x should have size [32, 1, 20, 6]')
            x = torch.stack(
                [
                    torch.stack(
                        [
                            torch.flatten(
                                self._patch_to_function.forward(
                                    torch.flatten(z, start_dim=-2, end_dim=-1)
                                ).to(device)
                            )
                            for z in torch.unbind(y)
                        ]
                    )
                    for y in torch.unbind(x)
                ]
            )
            print(f'After applying patchtofunction, x has size {x.size()}')
            return x
