import torch
import numpy as np
import json
import h5py
import tqdm
from torch.utils.data import Dataset

import itertools


from firedrake import (
    FunctionSpace,
    Function,
    SpatialCoordinate,
    UnitIcosahedralSphereMesh,
)

__all__ = [
    "show_hdf5_header",
    "load_hdf5_dataset",
    "SphericalFunctionSpaceDataset",
    "SolidBodyRotationDataset",
]


def load_hdf5_dataset(filename):
    """Load the dataset from disk

    :arg filename: name of file to load from"""
    with h5py.File(filename, "r") as f:
        data = f["base/data"]
        t_final = f["base/t_final"]
        metadata = json.loads(f["base/metadata"][()])
        dataset = SphericalFunctionSpaceDataset(
            int(f.attrs["n_func_in_dynamic"]),
            int(f.attrs["n_func_in_ancillary"]),
            int(f.attrs["n_func_target"]),
            int(f.attrs["n_ref"]),
            int(f.attrs["n_samples"]),
            data=np.asarray(data),
            t_final=np.asarray(t_final),
            metadata=metadata,
        )
    return dataset


def show_hdf5_header(filename):
    """Show the header of a hdf5 file

    :arg filename: name of file to inspect
    """
    print(f"header of {filename}")
    with h5py.File(filename, "r") as f:
        print("  attributes:")
        item = "class"
        print(f"    {item:20s} = {f.attrs[item]:20s}")
        for item in [
            "n_func_in_dynamic",
            "n_func_in_ancillary",
            "n_func_target",
            "n_ref",
            "n_dof",
            "n_samples",
        ]:
            print(f"    {item:20s} = {f.attrs[item]:8d}")
        print("  metadata:")
        metadata = json.loads(f["base/metadata"][()])
        for key, value in metadata.items():
            print(f"    {str(key):20s} = {str(value):20s}")


class SphericalFunctionSpaceDataset(Dataset):
    """Abstract base class for data generation on a function space
    defined on a spherical mesh

    yields input,target pairs ((X,t),y) where the input X is a tensor
    of shape (n_func, n_dof), t is a scalar and the target y is a tensor
    of shape (n_func_target,n_dof).
    """

    def __init__(
        self,
        n_func_in_dynamic,
        n_func_in_ancillary,
        n_func_target,
        n_ref,
        nsamples,
        data=None,
        t_final=None,
        metadata=None,
        dtype=None,
    ):
        """Initialise new instance

        :arg n_func_in_dynamic: number of dynamic input funtions
        :arg n_func_in_ancillary number of ancillary input functions
        :arg n_func_target: number of output functions
        :arg n_ref: number of mesh refinements
        :arg nsamples: number of samples
        :arg data: data to initialise with
        :arg t_final: final times
        :arg metadata: metadata to initialise with
        :arg dtype: type to which the data is converted to
        """
        self.n_func_in_dynamic = n_func_in_dynamic
        self.n_func_in_ancillary = n_func_in_ancillary
        self.n_func_target = n_func_target
        self.n_ref = n_ref
        self.dtype = torch.get_default_dtype() if dtype is None else dtype
        mesh = UnitIcosahedralSphereMesh(n_ref)  # create the mesh
        self._fs = FunctionSpace(mesh, "CG", 1)  # define the function space
        self.n_samples = nsamples
        self._data = (
            np.empty(
                (
                    self.n_samples,
                    self.n_func_in_dynamic
                    + self.n_func_in_ancillary
                    + self.n_func_target,
                    self._fs.dof_count,
                ),
                dtype=np.float64,
            )
            if data is None
            else data
        )
        self._t_final = (
            np.empty(self.n_samples, dtype=np.float64) if t_final is None else t_final
        )

        self.metadata = {} if metadata is None else metadata

    def __getitem__(self, idx):
        """Return a single sample (X,y)

        :arg idx: index of sample
        """
        X = torch.tensor(
            self._data[idx, : self.n_func_in_dynamic + self.n_func_in_ancillary],
            dtype=self.dtype,
        )
        t = torch.tensor(
            self._t_final[idx],
            dtype=self.dtype,
        )
        y = torch.tensor(
            self._data[idx, self.n_func_in_dynamic + self.n_func_in_ancillary :],
            dtype=self.dtype,
        )
        return (X, t), y

    def __len__(self):
        """Return numnber of samples"""
        return self.n_samples

    def save(self, filename):
        """Save the dataset to disk

        :arg filename: name of file to save to"""
        with h5py.File(filename, "w") as f:
            group = f.create_group("base")
            group.create_dataset("data", data=self._data)
            group.create_dataset("t_final", data=self._t_final)
            f.attrs["n_func_in_dynamic"] = int(self.n_func_in_dynamic)
            f.attrs["n_func_in_ancillary"] = int(self.n_func_in_ancillary)
            f.attrs["n_func_target"] = int(self.n_func_target)
            f.attrs["n_ref"] = int(self.n_ref)
            f.attrs["n_dof"] = int(self._fs.dof_count)
            f.attrs["n_samples"] = int(self.n_samples)
            f.attrs["class"] = type(self).__name__
            group.create_dataset("metadata", data=json.dumps(self.metadata))


class SolidBodyRotationDataset(SphericalFunctionSpaceDataset):
    """Data set for advection

    The input conists of the function fields (u,x,y,z) which represent a
    scalar function u and the three coordinate fields. The output is
    the same function, but rotated by some angle phi

    """

    def __init__(self, nref, nsamples, omega, t_final_max=1.0, degree=4, seed=12345):
        """Initialise new instance

        :arg nref: number of mesh refinements
        :arg nsamples: number of samples
        :arg omega: rotation speed
        :arg t_final_max: maximum final time
        :arg degree: polynomial degree used for generating random fields
        :arg seed: seed of rng
        """
        n_func_in_dynamic = 4
        n_func_in_ancillary = 3
        n_func_target = 1
        super().__init__(
            n_func_in_dynamic, n_func_in_ancillary, n_func_target, nref, nsamples
        )
        self.metadata = {
            "omega": f"{omega:}",
            "t_final_max": f"{t_final_max:}",
            "degree": degree,
            "seed": seed,
        }
        x, y, z = SpatialCoordinate(self._fs.mesh())
        self._u_x = Function(self._fs).interpolate(x)
        self._u_y = Function(self._fs).interpolate(y)
        self._u_z = Function(self._fs).interpolate(z)
        self._u = Function(self._fs)
        self._omega = omega
        self._t_final_max = t_final_max
        self._degree = degree
        self._rng = np.random.default_rng(
            seed
        )  # removing the seed seems to make it slower

    def generate(self):
        """Generate the data"""
        # generate data
        x, y, z = SpatialCoordinate(self._fs.mesh())
        for j in tqdm.tqdm(range(self.n_samples)):
            t_final = self._t_final_max * self._rng.uniform(0, 1)
            phi = self._omega * t_final
            expr_in = 0
            expr_in_dx = 0
            expr_in_dy = 0
            expr_in_dz = 0
            expr_target = 0
            coeff = self._rng.normal(size=(self._degree, self._degree, self._degree))
            for jx, jy, jz in itertools.product(
                range(self._degree), range(self._degree), range(self._degree)
            ):
                expr_in += coeff[jx, jy, jz] * x**jx * y**jy * z**jz
                if jx > 0:
                    expr_in_dx += coeff[jx, jy, jz] * jx * x ** (jx - 1) * y**jy * z**jz
                if jy > 0:
                    expr_in_dy += coeff[jx, jy, jz] * jy * x**jx * y ** (jy - 1) * z**jz
                if jz > 0:
                    expr_in_dz += coeff[jx, jy, jz] * jz * x**jx * y**jy * z ** (jz - 1)
                expr_target += (
                    coeff[jx, jy, jz]
                    * (x * np.cos(phi) - y * np.sin(phi)) ** jx
                    * (x * np.sin(phi) + y * np.cos(phi)) ** jy
                    * z**jz
                )
            self._u.interpolate(expr_in)
            self._data[j, 0, :] = self._u.dat.data
            self._u.interpolate(expr_in_dx)
            self._data[j, 1, :] = self._u.dat.data
            self._u.interpolate(expr_in_dy)
            self._data[j, 2, :] = self._u.dat.data
            self._u.interpolate(expr_in_dz)
            self._data[j, 3, :] = self._u.dat.data
            self._data[j, 4, :] = self._u_x.dat.data
            self._data[j, 5, :] = self._u_y.dat.data
            self._data[j, 6, :] = self._u_z.dat.data
            self._u.interpolate(expr_target)
            self._data[j, 7, :] = self._u.dat.data
            self._t_final[j] = t_final
