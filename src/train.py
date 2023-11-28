import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from firedrake import (
    UnitIcosahedralSphereMesh,
    Function,
    FunctionSpace,
    SpatialCoordinate,
)

# NB: need to import tensorflow *after* firedrake ...
import tensorflow as tf

from spherical_patch_covering import SphericalPatchCovering
from patch_interpolation import FunctionToPatchInterpolationLayer
from patch_encoder import PatchEncoder
from patch_decoder import PatchDecoder


############################################################################
# M A I N (for testing)
############################################################################

if __name__ == "__main__":
    spherical_patch_covering = SphericalPatchCovering(0, 4)

    print(f"number of patches               = {spherical_patch_covering.n_patches}")
    print(f"patchsize                       = {spherical_patch_covering.patch_size}")
    print(f"number of points in all patches = {spherical_patch_covering.n_points}")

    mesh = UnitIcosahedralSphereMesh(1)
    V = FunctionSpace(mesh, "CG", 1)

    layer = FunctionToPatchInterpolationLayer(V, spherical_patch_covering)

    u = Function(V)
    u.dat.data[2] = 1
    x, y, z = SpatialCoordinate(mesh)

    u_x = Function(V).interpolate(x)
    u_y = Function(V).interpolate(y)
    u_z = Function(V).interpolate(z)
    input = tf.expand_dims(
        tf.constant(
            [u.dat.data, u_x.dat.data, u_y.dat.data, u_z.dat.data], dtype=tf.float32
        ),
        axis=0,
    )

    encoder = PatchEncoder(n_dynamic=3, latent_dim=17, ancillary_dim=6)
    decoder = PatchDecoder(n_func=2, patch_size=spherical_patch_covering.patch_size)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(input)
        interp = layer(input)
        print(interp.shape)
        latent = encoder(interp)
        print(latent.shape)
        decoded = decoder(latent)
        print(decoded.shape)
