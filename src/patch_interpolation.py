import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from firedrake import (
    UnitIcosahedralSphereMesh,
    Function,
    FunctionSpace,
    VertexOnlyMesh,
    Interpolator,
    TestFunction,
)

# NB: need to import tensorflow *after* firedrake ...
import tensorflow as tf

from spherical_patch_covering import SphericalPatchCovering


class FunctionToPatchInterpolationLayer(tf.keras.layers.Layer):
    """Class for differentiable interpolation from a function space tensor
    to a spherical patch covering

    The input X has shape

        ([B],n_{func},n_{dof})

    where the batch-dimension of size B is optional. n_{func} is the number of
    functions to be interpolated and n_{dof} is the number of unknowns per function,
    as defined by the function space. The output has shape

        ([B],n_{patches},n_{func},n_{dof per patch})

    where n_{patches} and n_{dof per patch} depend on the SphericalPatchCovering
    """

    def __init__(self, fs, spherical_patch_covering):
        """Initialise new instance

        :arg fs: function space
        :arg patch_covering: spherical patch covering to interpolate onto
        """
        super().__init__()
        mesh = fs.mesh()
        self._spherical_patch_covering = spherical_patch_covering
        points = self._spherical_patch_covering.points.reshape(
            (self._spherical_patch_covering.n_points, 3)
        )
        vertex_only_mesh = VertexOnlyMesh(mesh, points)
        vertex_only_fs = FunctionSpace(vertex_only_mesh, "DG", 0)
        self._interpolator = Interpolator(TestFunction(fs), vertex_only_fs)
        self._u = Function(fs)
        self._w = Function(fs)
        self._cofunction_v = Function(vertex_only_fs.dual())

    @tf.custom_gradient
    def _interpolate(
        self,
        x,
    ):
        """Differentiable interpolation from a single dof-vector to all patches in
        the spherical patch covering

        Returns a tensor of shape (n_{patches},n_{points per patch})

        :arg x: input vector x of size n_{dof}
        """
        npatches = self._spherical_patch_covering.n_patches
        # Copy inputs to function
        with self._u.dat.vec as u_vec:
            u_vec[:] = x[:]
        # interpolate
        self._interpolator.interpolate(self._u, output=self._cofunction_v)
        # copy back to tensor
        with self._cofunction_v.dat.vec_ro as v_vec:
            y = tf.reshape(tf.convert_to_tensor(v_vec[:]), [npatches, -1])

        def grad(upstream):
            """Evaluate gradient

            :arg upstream: upstream gradient
            """
            with self._cofunction_v.dat.vec as v_vec:
                v_vec[:] = tf.reshape(upstream, [-1])[:]
            self._interpolator.interpolate(
                self._cofunction_v,
                output=self._w,
                transpose=True,
            )
            with self._w.dat.vec_ro as w_vec:
                grad_y = tf.convert_to_tensor(w_vec[:])
            return grad_y

        return y, grad

    def call(self, inputs):
        """Call layer for a given input tensor

        :arg inputs: tensor of shape ([B],n_{func},n_{dof})
        """
        input_dim = len(inputs.shape)
        if input_dim == 2:
            # add batch dimension if this is not already present
            X = tf.expand_dims(inputs, axis=0)
        elif input_dim == 3:
            X = inputs
        else:
            print(f"invalid input shape : {input_dim}")
        Y = tf.stack(
            [
                tf.stack(v, axis=0)
                for v in [
                    [self._interpolate(u) for u in tf.unstack(batch)]
                    for batch in tf.unstack(X)
                ]
            ],
            axis=0,
        )
        # Swap axes
        Y = tf.transpose(Y, perm=(0, 2, 1, 3))
        # u_interpolated now has shape (B,n_{func},n_{patches},n_{dof per patch})
        if input_dim == 2:
            # remove batch dimension again
            Y = tf.squeeze(Y, axis=0)
        return Y


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

    u1 = Function(V)
    u1.dat.data[2] = 1
    u2 = Function(V)
    u2.dat.data[6] = 1
    x = tf.expand_dims(
        tf.constant([u1.dat.data, u2.dat.data], dtype=tf.float64), axis=0
    )

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = layer.call(x)
        print(y.shape)
