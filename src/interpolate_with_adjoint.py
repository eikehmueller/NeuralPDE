import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from firedrake import *
import tensorflow as tf


class Patch:
    def __init__(self, points):
        pass
        self.points = points


class FunctionToPatchInterpolationLayer(tf.keras.layers.Layer):
    """Class for differentiable interpolation from a function space tensor
    to a set of patches"""

    def __init__(self, fs, patches):
        super().__init__()
        mesh = fs.mesh()
        self.npatches = len(patches)
        nodes = []
        for patch in patches:
            nodes += patch.points
        vertex_only_mesh = VertexOnlyMesh(mesh, nodes)
        vertex_only_fs = FunctionSpace(vertex_only_mesh, "DG", 0)
        self._interpolator = Interpolator(TestFunction(fs), vertex_only_fs)
        self._u = Function(fs)
        self._cofunction_v = Function(vertex_only_fs.dual())

    @tf.custom_gradient
    def _call(self, x):
        """Differentiable interpolation

        :arg x: input vector x
        """
        # Copy inputs to function
        with self._u.dat.vec as u_vec:
            u_vec[:] = x[:]
        # interpolate
        v = self._interpolator.interpolate(self._u, output=self._cofunction_v)
        # copy back to tensor
        with self._cofunction_v.dat.vec_ro as v_vec:
            y = tf.reshape(tf.convert_to_tensor(v_vec[:]), [self.npatches, -1])

        def grad(upstream):
            """Evaluate gradient"""
            with self._cofunction_v.dat.vec as v_vec:
                v_vec[:] = upstream[:]
            w = self._interpolator.interpolate(self._cofunction_v, transpose=True)
            with w.dat.vec_ro as w_vec:
                grad_y = tf.reshape(tf.convert_to_tensor(w_vec[:]), [self.npatches, -1])
            return grad_y

        return y, grad

    def call(self, inputs):
        input_shape = len(inputs.shape)
        if input_shape == 1:
            return self._call(inputs)
        elif input_shape == 2:
            return tf.stack([self._call(x) for x in tf.unstack(inputs)])
        else:
            print(f"invalid input shape : {input_shape}")


mesh = IcosahedralSphereMesh(1)
V = FunctionSpace(mesh, "CG", 1)

patches = [
    Patch([[0, 0, 1], [0, 0, 1], [0, 0, 1]]),
    Patch([[0, 0, -1], [0, 0, -1], [0, 0, -1]]),
]
layer = FunctionToPatchInterpolationLayer(V, patches)

u1 = Function(V)
u1.dat.data[2] = 1
u2 = Function(V)
u2.dat.data[6] = 1
x = tf.constant([u1.dat.data, u2.dat.data], dtype=tf.float64)
with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    y = layer.call(x)
    print(y)
    print(tape.gradient(y, x))
