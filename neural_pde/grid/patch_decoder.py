"""Patch decoder. Decodes information from latent space back to a function on the sphere"""

from firedrake import *
import torch
from neural_pde.grid.intergrid import AdjointInterpolator


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

    def __init__(self, fs, spherical_patch_covering, decoder_model, dtype=None):
        """Initialise instance

        :arg fs: function space
        :arg spherical_patch_covering: the patch covering for the intermediate
            stage
        :arg decoder_model: model that maps tensors of shape
            (d_{lat},) to tensors of shape (n_{out},patch_size)
        :arg dtype: datatype. Use torch default if None
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
        self._patch_to_function = AdjointInterpolator(
            fs,
            vertex_only_fs,
            dtype=torch.get_default_dtype() if dtype is None else dtype,
        )

    def forward(self, x):
        """Forward map

        Returns tensor of shape (B,n_func,n_output)

        :arg x: input, tensor of shape (B,n_patches,d_dynamic+d_ancillary)
        """
        # Part I: encoding on patches
        # x has shape (B,n_patches,d_dynamic+d_ancillary)
        x = self._decoder_model(x)
        # now x has shape (B,n_patches,patchsize,n_output)
        dim = x.dim()
        # Part II: permutation
        # permutation idx = [0,1,...,d-4,d-2,d-3,d-1] for example:
        # d = 3: idx = [1,0,2], d = 4: idx = [0,2,1,3], d = 5: idx = [0,1,3,2,4]
        idx = list(range(dim - 3)) + [dim - 2, dim - 3, dim - 1]
        x = torch.permute(x, idx)
        # now x has shape (B,n_output,n_patches,patchsize)
        # Part III: (adjoint) interpolation from VOM to spherical function space
        x = self._patch_to_function.forward(torch.flatten(x, start_dim=-2, end_dim=-1))
        # now x has shape (B,n_output,n_dof)
        return x
