import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

import itertools

from abc import ABC, abstractmethod

from firedrake import (
    UnitIcosahedralSphereMesh,
    Function,
    VTKFile,
    FunctionSpace,
    SpatialCoordinate,
)

import argparse


class SphericalFunctionSpaceDataset(ABC, Dataset):
    """Abstract base class for data generation on a function space
    defined on a spherical mesh

    yields input,target pairs (X,y) where the input X is a tensor
    of shape (n_func, n_dof) and the target y is a tensor
    of shape (n_func_target,n_dof).
    """

    def __init__(self, fs, nsamples):
        """Initialise new instance

        :arg fs: function space
        """
        self._fs = fs
        self._nsamples = nsamples
        self._data = np.empty(
            (self._nsamples, self.n_func_in + self.n_func_target, self._fs.dof_count),
            dtype=np.float64,
        )

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

    @abstractmethod
    def __getitem__(self, idx):
        """Return a single sample (X,y)

        :arg idx: index of sample
        """
        pass

    def __len__(self):
        """Return numnber of samples"""
        return self._nsamples

    def save(self, filename):
        """Save the dataset to disk

        :arg filename: name of file to save to"""
        np.save(filename, self._data)

    def load(self, filename):
        """Load the dataset from disk

        :arg filename: name of file to load from"""
        self._data = np.load(filename)
        assert self._data.shape == (
            self._nsamples,
            self.n_func_in + self.n_func_target,
            self._fs.dof_count,
        )


class AdvectionDataset(SphericalFunctionSpaceDataset):
    """Data set for advection

    The input conists of the function fields (u,x,y,z) which represent a
    scalar function u and the three coordinate fields. The output is
    the same function, but rotated by some angle phi

    """

    def __init__(self, fs, nsamples, phi, degree=4, seed=12345):
        """Initialise new instance

        :arg fs: function space
        :arg phi: rotation angle phi
        :arg degree: polynomial degree used for generating random fields
        """
        self._n_func_in = 4
        self._n_func_target = 1
        super().__init__(fs, nsamples)
        mesh = self._fs.mesh()
        x, y, z = SpatialCoordinate(mesh)
        self._u_x = Function(self._fs).interpolate(x)
        self._u_y = Function(self._fs).interpolate(y)
        self._u_z = Function(self._fs).interpolate(z)
        self._u = Function(self._fs)
        self._phi = phi
        self._degree = degree
        self._rng = np.random.default_rng(
            seed
        )  # removing the seed seems to make it slower

    def generate(self):
        """Generate the data"""
        # generate data
        x, y, z = SpatialCoordinate(self._fs.mesh())
        for j in range(self._nsamples):
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
            self._u.interpolate(expr_in)
            self._data[j, 0, :] = self._u.dat.data
            self._data[j, 1, :] = self._u_x.dat.data
            self._data[j, 2, :] = self._u_y.dat.data
            self._data[j, 3, :] = self._u_z.dat.data
            self._u.interpolate(expr_target)
            self._data[j, 4, :] = self._u.dat.data

    def __getitem__(self, idx):
        """Return a single sample (X,y)

        :arg idx: index of sample
        """
        X = torch.tensor(self._data[idx, : self._n_func_in], dtype=torch.float64)
        y = torch.tensor(self._data[idx, self._n_func_in :], dtype=torch.float64)
        return (X, y)


#######################################################################
# M A I N
#######################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    default_path = "../output"
    parser.add_argument(
        "--path_to_output_folder",
        type=str,
        action="store",
        default=default_path,
        help="path to output folder",
    )
    args = parser.parse_args()
    path_to_output = args.path_to_output_folder

    mesh = UnitIcosahedralSphereMesh(3)
    V = FunctionSpace(mesh, "CG", 1)
    degree = 4
    nsamples = 32
    batchsize = 4

    dataset = AdvectionDataset(V, nsamples, degree)

    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    for k, batched_sample in enumerate(iter(dataloader)):
        print(k, batched_sample[0].shape, batched_sample[1].shape)
        u_in = Function(V, name="input")
        u_in.dat.data[:] = batched_sample[0][0, 0].numpy()
        u_target = Function(V, name="target")
        u_target.dat.data[:] = batched_sample[1][0].numpy()
        file = VTKFile(f"{path_to_output}/sample.pvd")
        file.write(u_in, u_target)

