import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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
    def __init__(self, fs):
        self._fs = fs

    @property
    def n_dof(self):
        """Number of unknowns"""
        return len(Function(self._fs).dat.data)

    @property
    def n_func(self):
        """Number of input functions (including auxilliary functions)"""
        return self._n_func

    @property
    def n_func_target(self):
        """Number of functions in the target"""
        return self._n_func_target

    @abstractmethod
    def __call__(self):
        pass


class AdvectionDataGenerator(DataGenerator):
    def __init__(self, fs):
        super().__init__(fs)
        mesh = self._fs.mesh()
        x, y, z = SpatialCoordinate(mesh)
        self._u_x = Function(self._fs).interpolate(x)
        self._u_y = Function(self._fs).interpolate(y)
        self._u_z = Function(self._fs).interpolate(z)
        self._u = Function(self._fs)

        self._n_func = 4
        self._n_func_target = 1

    def __call__(self):
        while True:
            X = tf.constant(
                [
                    self._u.dat.data,
                    self._u_x.dat.data,
                    self._u_y.dat.data,
                    self._u_z.dat.data,
                ],
                dtype=tf.float32,
            )
            y = tf.constant(
                [
                    self._u.dat.data,
                ],
                dtype=tf.float32,
            )
            yield (X, y)


if __name__ == "__main__":
    mesh = UnitIcosahedralSphereMesh(1)
    V = FunctionSpace(mesh, "CG", 1)

    generator = AdvectionDataGenerator(V)
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            (
                tf.TensorSpec(
                    shape=(generator.n_func, generator.n_dof), dtype=tf.float32
                ),
                tf.TensorSpec(
                    shape=(generator.n_func_target, generator.n_dof), dtype=tf.float32
                ),
            )
        ),
    )
    for z in dataset.take(1):
        print(z[0].shape, z[1].shape)
