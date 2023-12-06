from firedrake import (
    Function,
    FunctionSpace,
    VertexOnlyMesh,
    Interpolator,
    TestFunction,
)

# NB: need to import tensorflow *after* firedrake ...
import tensorflow as tf


class FunctionToPatchInterpolationLayer(tf.keras.layers.Layer):
    """Class for differentiable interpolation from a function space tensor
    to a spherical patch covering

    The input X has shape

        (B,n_{func},n_{dof})

    where the batch-dimension is of size B. n_{func} is the number of
    functions to be interpolated and n_{dof} is the number of unknowns per function,
    as defined by the function space. The output has shape

        (B,n_{patches},n_{func},n_{dof per patch})

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
        self._v = Function(fs)
        self._cofunction_u = Function(vertex_only_fs.dual())
        self._cofunction_v = Function(vertex_only_fs.dual())

    @tf.custom_gradient
    def _interpolate(
        self,
        x,
    ):
        """Differentiable interpolation from a single dof-vector to all patches in
        the spherical patch covering

        Returns a tensor of shape (B,n_{patches},n_{points per patch})

        :arg x: input vector x of size n_{dof}
        """
        # Copy inputs to function
        if not tf.is_symbolic_tensor(x):
            with self._u.dat.vec as u_vec:
                u_vec[:] = x[:]
        # interpolate
        self._interpolator.interpolate(self._u, output=self._cofunction_v)
        # copy back to tensor
        with self._cofunction_v.dat.vec_ro as v_vec:
            y = tf.reshape(
                tf.convert_to_tensor(v_vec[:], dtype=tf.float32),
                [self._spherical_patch_covering.n_patches, -1],
            )

        def grad(upstream):
            """Evaluate gradient

            :arg upstream: upstream gradient
            """
            with self._cofunction_u.dat.vec as u_vec:
                u_vec[:] = tf.reshape(upstream, [-1])[:]
            self._interpolator.interpolate(
                self._cofunction_u,
                output=self._v,
                transpose=True,
            )
            with self._v.dat.vec_ro as v_vec:
                grad_y = tf.convert_to_tensor(v_vec[:], dtype=tf.float32)
            return grad_y

        return y, grad

    def call(self, inputs):
        """Call layer for a given input tensor

        :arg inputs: tensor of shape (B,n_{func},n_{dof})
        """
        return tf.transpose(
            tf.stack(
                [
                    tf.stack(v, axis=0)
                    for v in [
                        [self._interpolate(u) for u in tf.unstack(funcs)]
                        for funcs in tf.unstack(inputs)
                    ]
                ],
                axis=0,
            ),
            perm=(0, 2, 1, 3),
        )


class PatchToFunctionInterpolationLayer(tf.keras.layers.Layer):
    """Class for differentiable interpolation from the data on a spherical
    patch covering to a function space tensor

    The input X has shape

        (B,n_{patches},n_{func},n_{dof per patch})

    where the batch-dimension is of size B and n_{func} is the number of. n_{patches} and
    n_{dof per patch} depend on the SphericalPatchCovering
    functions to be interpolated. The output has shape

        (B,n_{func},n_{dof})

    where n_{dof} is the number of unknowns per function, as defined by the function space.
    """

    def __init__(self, spherical_patch_covering, fs):
        """Initialise new instance

        :arg patch_covering: spherical patch covering to interpolate onto
        :arg fs: function space
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
        self._v = Function(fs)
        self._cofunction_u = Function(vertex_only_fs.dual())
        self._cofunction_v = Function(vertex_only_fs.dual())

    @tf.custom_gradient
    def _interpolate(
        self,
        x,
    ):
        """Differentiable interpolation from all patches to a single dof-vector

        Returns a tensor of size n_{dof}

        :arg x: input vector x of shape (n_{patches},n_{points per patch})
        """
        # Copy inputs to function
        if not tf.is_symbolic_tensor(x):
            with self._cofunction_u.dat.vec as u_vec:
                u_vec[:] = tf.reshape(x, [-1])[:]
        # interpolate
        self._interpolator.interpolate(
            self._cofunction_u, output=self._v, transpose=True
        )
        # copy back to tensor
        with self._v.dat.vec_ro as v_vec:
            y = tf.convert_to_tensor(v_vec[:], dtype=tf.float32)

        def grad(upstream):
            """Evaluate gradient

            :arg upstream: upstream gradient
            """
            if not tf.is_symbolic_tensor(upstream):
                with self._u.dat.vec as u_vec:
                    u_vec[:] = upstream[:]
            self._interpolator.interpolate(
                self._u,
                output=self._cofunction_v,
            )
            with self._cofunction_v.dat.vec_ro as v_vec:
                grad_y = tf.reshape(
                    tf.convert_to_tensor(v_vec[:], dtype=tf.float32),
                    [self._spherical_patch_covering.n_patches, -1],
                )
            return grad_y

        return y, grad

    def call(self, inputs):
        """Call layer for a given input tensor

        :arg inputs: tensor of shape (B,n_{patches},n_{func},n_{dof per patch})
        """
        return tf.stack(
            [
                tf.stack(v, axis=0)
                for v in [
                    [self._interpolate(u) for u in tf.unstack(funcs)]
                    for funcs in tf.unstack(tf.transpose(inputs, perm=(0, 2, 1, 3)))
                ]
            ],
            axis=0,
        )
