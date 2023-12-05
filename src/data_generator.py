import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import itertools

from abc import ABC, abstractmethod

from firedrake import (
    UnitIcosahedralSphereMesh,
    Function,
    File,
    FunctionSpace,
    SpatialCoordinate,
)

# NB: need to import tensorflow *after* firedrake ...
import tensorflow as tf


class DataGenerator(ABC):
    """Abstract base class for data generation on a function space

    yields input,target pairs (X,y) where the input X is a tensor
    of shape (n_func, n_dof) and the target y is a tensor
    of shape (n_func_target,n_dof).
    """

    def __init__(self, fs):
        """Initialise new instance

        :arg fs: function space
        """
        self._fs = fs

    @property
    def n_dof(self):
        """Number of unknowns"""
        return len(Function(self._fs).dat.data)

    @property
    def n_func_in(self):
        """Number of input functions (including auxilliary functions)"""
        return self._n_func_in

    @property
    def n_func_target(self):
        """Number of functions in the target"""
        return self._n_func_target

    @property
    def output_signature(self):
        """Return output signature"""
        return (
            tf.TensorSpec(shape=(self.n_func_in, self.n_dof), dtype=tf.float32),
            tf.TensorSpec(shape=(self.n_func_target, self.n_dof), dtype=tf.float32),
        )

    @abstractmethod
    def __call__(self):
        """Return a single sample (X,y)"""
        pass


class AdvectionDataGenerator(DataGenerator):
    """Data generator for advection

    The input conists of the function fields (u,x,y,z) which represent a
    scalar function u and the three coordinate fields. The output is
    the same function, but rotated by some angle phi

    """

    def __init__(self, fs, phi, degree=4):
        """Initialise new instance

        :arg fs: function space
        :arg phi: rotation angle phi
        :arg degree: polynomial degree used for generating random fields
        """
        super().__init__(fs)
        mesh = self._fs.mesh()
        x, y, z = SpatialCoordinate(mesh)
        self._u_x = Function(self._fs).interpolate(x)
        self._u_y = Function(self._fs).interpolate(y)
        self._u_z = Function(self._fs).interpolate(z)
        self._u = Function(self._fs)
        self._phi = phi

        self._n_func_in = 4
        self._n_func_target = 1

        self._degree = degree
        self._rng = np.random.default_rng(12345)

    def __call__(self):
        """Return a single sample (X,y)"""
        while True:
            x, y, z = SpatialCoordinate(self._fs.mesh())
            expr_in = 0
            expr_target = 0
            coeff = self._rng.normal(size=(self._degree, self._degree, self._degree))
            for jx, jy, jz in itertools.product(
                range(self._degree), range(self._degree), range(self._degree)
            ):
                expr_in += coeff[jx, jy, jz] * x**jx * y**jy * z**jz
                expr_target += (
                    coeff[jx, jy, jz]
                    * (x * np.cos(self._phi) - y * np.sin(self._phi)) ** jx
                    * (x * np.sin(self._phi) + y * np.cos(self._phi)) ** jy
                    * z**jz
                )
            self._u.project(expr_in)
            X = tf.constant(
                [
                    self._u.dat.data,
                    self._u_x.dat.data,
                    self._u_y.dat.data,
                    self._u_z.dat.data,
                ],
                dtype=tf.float32,
            )
            self._u.project(expr_target)
            y = tf.constant(
                [
                    self._u.dat.data,
                ],
                dtype=tf.float32,
            )
            yield (X, y)


#######################################################################
# M A I N
#######################################################################
if __name__ == "__main__":
    mesh = UnitIcosahedralSphereMesh(3)
    V = FunctionSpace(mesh, "CG", 1)

    generator = AdvectionDataGenerator(V, 1)
    dataset = tf.data.Dataset.from_generator(
        generator, output_signature=generator.output_signature
    )
    for z in dataset.take(1):
        print(z[0].shape, z[1].shape)
        u_in = Function(V, name="input")
        u_in.dat.data[:] = z[0][0].numpy()
        u_target = Function(V, name="target")
        u_target.dat.data[:] = z[1][0].numpy()
        file = File("sample.pvd")
        file.write(u_in, u_target)
