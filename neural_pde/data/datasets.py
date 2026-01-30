import torch
import numpy as np
from scipy.stats import truncnorm
import json
import h5py
import tqdm
from torch.utils.data import Dataset
import itertools

try:
    from gusto import (
        lonlatr_from_xyz,
        ShallowWaterParameters,
        ShallowWaterEquations,
        Domain,
        OutputParameters,
        IO,
        SSPRK3,
        DGUpwind,
        SemiImplicitQuasiNewton,
        RelativeVorticity,
        Divergence,
        SubcyclingOptions,
        RungeKuttaFormulation,
        Sum,
    )
except:
    print("WARNING: running without gusto support")

import neural_pde.util.diagnostics as dg
from neural_pde.util.velocity_functions import Projector as Proj

from firedrake import (
    FunctionSpace,
    Function,
    SpatialCoordinate,
)
from firedrake import *
from timeit import default_timer as timer
from datetime import timedelta

try:
    from gusto import (
        lonlatr_from_xyz,
        ShallowWaterParameters,
        ShallowWaterEquations,
        Domain,
        OutputParameters,
        IO,
        SSPRK3,
        DGUpwind,
        SemiImplicitQuasiNewton,
        RelativeVorticity,
        Divergence,
        Perturbation,
    )
except:
    print("WARNING: unable to import gusto")

__all__ = [
    "show_hdf5_header",
    "load_hdf5_dataset",
    "SphericalFunctionSpaceDataset",
    "SolidBodyRotationDataset",
    "ShallowWaterEquationsDataset",
]


def load_hdf5_dataset(filename):
    """Load the dataset from disk

    :arg filename: name of file to load from"""
    with h5py.File(filename, "r") as f:
        data = f["base/data"]
        t_initial = f["base/t_initial"]
        t_final = f["base/t_final"]
        metadata = json.loads(f["base/metadata"][()])
        dataset = SphericalFunctionSpaceDataset(
            int(f.attrs["n_func_in_dynamic"]),
            int(f.attrs["n_func_in_ancillary"]),
            int(f.attrs["n_func_target"]),
            int(f.attrs["n_ref"]),
            int(f.attrs["n_samples"]),
            data=np.asarray(data),
            t_initial=np.asarray(t_initial),
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
        radius,
        n_ref,
        nsamples,
        data=None,
        t_initial=None,
        t_final=None,
        metadata=None,
        dtype=None,
    ):
        """Initialise new instance

        :arg n_func_in_dynamic: number of dynamic input funtions
        :arg n_func_in_ancillary number of ancillary input functions
        :arg n_func_target: number of output functions
        :arg radius: the radius of the sphere
        :arg n_ref: number of mesh refinements
        :arg nsamples: number of samples
        :arg data: data to initialise with
        :arg t_initial: initial time
        :arg t_final: final time
        :arg metadata: metadata to initialise with
        :arg dtype: type to which the data is converted to
        """
        self.n_func_in_dynamic = n_func_in_dynamic
        self.n_func_in_ancillary = n_func_in_ancillary
        self.n_func_target = n_func_target
        self.n_ref = n_ref
        self.radius = radius
        self.dtype = torch.get_default_dtype() if dtype is None else dtype
        self.mesh = IcosahedralSphereMesh(
            radius=self.radius, refinement_level=n_ref, name="IcosahedralMesh"
        )  # create the mesh
        self._fs = FunctionSpace(self.mesh, "CG", 1)  # define the function space
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
        self._t_initial = (
            np.empty(self.n_samples, dtype=np.float64)
            if t_initial is None
            else t_initial
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
            self._t_final[idx] - self._t_initial[idx],
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
            group.create_dataset("t_initial", data=self._t_initial)
            group.create_dataset("t_final", data=self._t_final)
            f.attrs["n_func_in_dynamic"] = int(self.n_func_in_dynamic)
            f.attrs["n_func_in_ancillary"] = int(self.n_func_in_ancillary)
            f.attrs["n_func_target"] = int(self.n_func_target)
            f.attrs["radius"] = float(self.radius)
            f.attrs["n_ref"] = int(self.n_ref)
            f.attrs["n_dof"] = int(self._fs.dof_count)
            f.attrs["n_samples"] = int(self.n_samples)
            f.attrs["class"] = type(self).__name__
            group.create_dataset("metadata", data=json.dumps(self.metadata))

    @property
    def mean(self):
        """Return mean of data by averaging"""
        return np.mean(self._data, axis=0)

    @property
    def std(self):
        """Return standard deviation of data averaging over batches"""
        return np.std(self._data, axis=0)


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
        radius = 1
        super().__init__(
            n_func_in_dynamic,
            n_func_in_ancillary,
            n_func_target,
            radius,
            nref,
            nsamples,
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
        self.nt = 1

    def generate(self):
        """Generate the data as a random polynomial over the sphere.
        Stores the gradients of the polynomial in the x, y, and z directions."""
        # generate data
        x, y, z = SpatialCoordinate(self._fs.mesh())
        for j in tqdm.tqdm(range(self.n_samples)):
            t_final = self._t_final_max * self._rng.uniform(low=0.0, high=1.0)
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
                expr_target += (
                    coeff[jx, jy, jz]
                    * (x * np.cos(phi) - y * np.sin(phi)) ** jx
                    * (x * np.sin(phi) + y * np.cos(phi)) ** jy
                    * z**jz
                )
                if jx > 0:
                    expr_in_dx += coeff[jx, jy, jz] * jx * x ** (jx - 1) * y**jy * z**jz
                if jy > 0:
                    expr_in_dy += coeff[jx, jy, jz] * jy * x**jx * y ** (jy - 1) * z**jz
                if jz > 0:
                    expr_in_dz += coeff[jx, jy, jz] * jz * x**jx * y**jy * z ** (jz - 1)

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
            self._t_initial[j] = 0.0
            self._t_final[j] = t_final


class ShallowWaterEquationsDataset(SphericalFunctionSpaceDataset):
    """Data set for Shallow Water Equations

    The data is generated using the firedrake package gusto. The height h,
    and the fluid speed u in the x, y, and z directions are saved.
    The fields are normalised in h and u but are proportional to the
    standard measurements for oceans on the Earth.

    """

    def __init__(
        self,
        n_ref,
        nsamples,
        dt,
        t_final_max=10.0,
        omega=7.292e-5,
        g=9.8,
        radius=1,
        t_interval=10,
        t_sigma=1,
        t_lowest=0,
        t_highest=10,
        save_diagnostics=False,
    ):
        """Initialise new instance

        :arg nref: number of mesh refinements
        :arg nsamples: number of samples
        :arg dt: length of a timestep
        :arg t_final_max: maximum final time
        :arg omega: rotation speed - angular rotation of the earth
        :arg g: gravitational acceleration on the earth
        :arg t_interval: expected value of a sampled time interval
        :arg t_sigma: standard deviation from end time of t_interval
        :arg t_lowest: lowest possible sampled time (used to split training
                       and test data)
        :arg t_highest: highest possible sampled time (used to split training
                       and test data)
        :arg save_diagnostics: whether to save gusto's divergence and vorticity
        """

        n_func_in_dynamic = 3  # height, vorticity and divergence
        n_func_in_ancillary = 3  # x-, y-, and z- coordinates
        n_func_target = 3  # height, vorticity and divergence

        # initialise with the SphericalFunctionSpaceData
        super().__init__(
            n_func_in_dynamic,
            n_func_in_ancillary,
            n_func_target,
            radius,
            n_ref,
            nsamples,
        )

        self.omega = omega
        self.g = g
        self.radius = radius
        self.save_diagnostics = save_diagnostics

        self.metadata = {
            "omega": f"{self.omega:}",
            "t_final_max": f"{t_final_max:}",
            "t_lowest": f"{t_lowest}",
            "t_highest": f"{t_highest}",
        }
        self.dt = dt  # number of timesteps
        self.t_final_max = t_final_max  # final time
        self.t_interval = t_interval  # the expected
        self.t_lowest = t_lowest
        self.t_highest = t_highest
        self.t_sigma = t_sigma

        x, y, z = SpatialCoordinate(self._fs.mesh())  # spatial coordinate
        self._x = Function(self._fs).interpolate(x)  # collect data on x,y,z coordinates
        self._y = Function(self._fs).interpolate(y)
        self._z = Function(self._fs).interpolate(z)

    def generate_full_dataset(self):
        """
        Generate the full data for the shallow water equations using gusto.
        The dataset used to train the model is sampled from this data.
        """

        start_timer = timer()

        radius = 6371220.0  # planetary radius (m)
        mean_depth = 5960  # reference depth (m)
        g = 9.80616  # acceleration due to gravity (m/s^2)
        u_max = 20.0  # max amplitude of the zonal wind (m/s)
        mountain_height = 3000.0  # height of mountain (m)
        R0 = pi / 9.0  # radius of mountain (rad)
        lamda_c = -pi / 2.0  # longitudinal centre of mountain (rad)
        phi_c = pi / 6.0  # latitudinal centre of mountain (rad)

        # THE PROBLEM IS THE MESH

        element_order = 1  # CG method
        ncells_per_edge = 16
        # self.mesh = GeneralIcosahedralSphereMesh(radius, ncells_per_edge, degree=1)
        # self.mesh =IcosahedralSphereMesh(radius=radius, refinement_level=3)
        domain = Domain(self.mesh, self.dt, "BDM", element_order)

        x, y, z = SpatialCoordinate(self.mesh)
        lamda, phi, _ = lonlatr_from_xyz(x, y, z)  # latitide and longitude

        # mountain parameters
        rsq = min_value(R0**2, (lamda - lamda_c) ** 2 + (phi - phi_c) ** 2)
        r = sqrt(rsq)
        tpexpr = mountain_height * (1 - r / R0)

        # initialise parameters object
        parameters = ShallowWaterParameters(
            self.mesh, H=mean_depth, g=g, topog_expr=tpexpr
        )

        # set up the finite element form
        eqns = ShallowWaterEquations(domain, parameters)

        # BDM means Brezzi-Douglas-Marini finite element basis function

        # ShallowWaterParameters are the physical parameters for the shallow water equations
        # mean_depth = 1  # this is the parameter we nondimensionalise around
        # g0 = self.g * (T0**2) / L0  # nondimensionalised g (m/s^2)
        # Omega0 = self.omega * T0  # nondimensionalised omega (s^-1)

        # output options
        output = OutputParameters(
            dirname="gusto_output",
            dump_vtus=True,
            dump_nc=True,
            dump_diagnostics=True,
            dumpfreq=1,
            dumplist=["D", "topography"],
            checkpoint=True,
            chkptfreq=1,
            multichkpt=True,
        )  # these have been modified so we get no output

        # choose which fields to record over the simulation
        diagnostic_fields = [
            Sum("D", "topography"),
            Divergence("u"),
            RelativeVorticity(),
        ]
        io = IO(domain, output, diagnostic_fields=diagnostic_fields)

        subcycling_options = SubcyclingOptions(subcycle_by_courant=0.25)
        transported_fields = [
            SSPRK3(domain, "u", subcycling_options=subcycling_options),
            SSPRK3(
                domain,
                "D",
                subcycling_options=subcycling_options,
                rk_formulation=RungeKuttaFormulation.linear,
            ),
        ]
        transport_methods = [
            DGUpwind(eqns, "u"),
            DGUpwind(eqns, "D", advective_then_flux=True),
        ]
        stepper = SemiImplicitQuasiNewton(
            eqns, io, transported_fields, transport_methods, tau_values={"D": 1.0}
        )

        # setting the initial conditions for velocity
        u0 = stepper.fields("u")
        # day = 24 * 60 * 60 #/ T0  # day in seconds
        # u_max = 2 * pi * R / (12 * day)  # Maximum amplitude of the zonal wind (m/s)
        # u_max = 2 * pi / (12 * day)  # Maximum amplitude of the zonal wind (m/s)
        # uexpr = as_vector(
        #    (-u_max * cos(lat) * sin(lon), u_max * cos(lat) * cos(lon), 0)
        #

        # setting up initial conditions for height
        D0 = stepper.fields("D")
        g = parameters.g
        H = parameters.H
        Omega = parameters.Omega

        # adding mountain ranges - these could all be varied!
        # lamda_c = -pi / 4.0  # longitudinal centre of mountain (rad)
        # phi_c = - pi / 3.0  # latitudinal centre of mountain (rad)

        # R0 = pi / 9.0  # radius of mountain (rad)
        # mountain_height = 500 / L0  # height of mountain (m)

        # rsq1 = min_value(R0**2, (lon - lamda_c) ** 2 + (lat - phi_c) ** 2)

        # r1 = sqrt(rsq1)

        # tpexpr = mountain_height * ((1 - r1 / R0))
        # Dexpr = (
        #    H - ((R * Omega0 * u_max + 0.5 * u_max**2) * (sin(lat)) ** 2) / g# + tpexpr
        # )

        uexpr = as_vector([-u_max * y / radius, u_max * x / radius, 0.0])
        Dexpr = (
            mean_depth
            - tpexpr
            - (radius * Omega * u_max + 0.5 * u_max**2) * (z / radius) ** 2 / g
        )
        u0.project(uexpr)
        D0.interpolate(Dexpr)

        # reference velocity is zero, reference depth is H
        Dbar = Function(D0.function_space()).assign(mean_depth)
        stepper.set_reference_profiles([("D", Dbar)])

        # run the simulation
        stepper.run(t=0, tmax=self.t_final_max)
        end_timer = timer()
        print(f"Gusto runtime: {timedelta(seconds=end_timer-start_timer)}")

    def prepare_for_model(self, filename):
        """Sample the data from the generated dataset and save as a np array

        :arg filename: name of checkpoint file to read from
        """
        start_timer1 = timer()

        print(f"Filename is {filename}")

        with CheckpointFile(filename, "r") as afile:
            print("we have opened the checkpoint file")
            mesh_h5 = afile.load_mesh("IcosahedralMesh")
            V_BDM = FunctionSpace(mesh_h5, "BDM", 2)
            V_DG = FunctionSpace(mesh_h5, "DG", 1)
            V_CG = FunctionSpace(mesh_h5, "CG", 1)

            h_inp = Function(V_CG)  # input function for h
            h_tar = Function(V_CG)  # target function for h

            p1 = Proj(V_BDM, V_CG)
            p2 = Proj(V_DG, V_CG)

            nt = int(self.t_final_max / self.dt)

            diagnostics = dg.Diagnostics(V_BDM, V_CG)
            if self.save_diagnostics:
                file = VTKFile("results/gusto_output/diagnostics.pvd")

                for i in range(nt):
                    u_func = [Function(V_CG) for _ in range(3)]
                    u = afile.load_function(mesh_h5, "u", idx=i)
                    p1.apply(u, u_func)  # input u data

                    vorticity = diagnostics.vorticity(u)
                    divergence = diagnostics.divergence(u)
                    vorticity.rename("vorticity")
                    divergence.rename("divergence")
                    file.write(vorticity, divergence, u, t=i)

            interval = self.t_interval / self.dt
            sigma = self.t_sigma / self.dt

            for j in tqdm.tqdm(range(self.n_samples)):

                # randomly sample the generated data

                highest = self.t_highest / self.dt
                lowest = (
                    self.t_lowest / self.dt + 1
                )  # there is an error in gusto at t = 0 in the divergence
                start = np.random.randint(lowest, highest)

                if np.isclose(0, interval):
                    end = start
                else:
                    mu = start + interval
                    t_norm = truncnorm(
                        (start - mu) / sigma,
                        (highest - mu) / sigma,
                        loc=mu,
                        scale=sigma,
                    )
                    end = round(t_norm.rvs(1)[0])

                if j == 100:
                    print(f"Start timestep is {start}")
                    print(f"End timestep is {end}")

                if end > highest:
                    end = highest

                w1 = afile.load_function(mesh_h5, "u", idx=start)
                w2 = afile.load_function(mesh_h5, "u", idx=end)

                h1 = afile.load_function(mesh_h5, "D", idx=start)
                h2 = afile.load_function(mesh_h5, "D", idx=end)

                p2.apply(h1, h_inp)
                p2.apply(h2, h_tar)

                vorticity_inp = diagnostics.vorticity(w1)
                divergence_inp = diagnostics.divergence(w1)
                vorticity_tar = diagnostics.vorticity(w2)
                divergence_tar = diagnostics.divergence(w2)

                # input data - dynamic variables
                self._data[j, 0, :] = h_inp.dat.data  # h data, take away mean depth
                self._data[j, 1, :] = divergence_inp.dat.data
                self._data[j, 2, :] = vorticity_inp.dat.data
                # coordinate data - auxiliary variables
                self._data[j, 3, :] = self._x.dat.data  # x coord data
                self._data[j, 4, :] = self._y.dat.data  # y coord data
                self._data[j, 5, :] = self._z.dat.data  # z coord data
                # output data - target data
                self._data[j, 6, :] = h_tar.dat.data  # h data
                self._data[j, 7, :] = divergence_tar.dat.data
                self._data[j, 8, :] = vorticity_tar.dat.data
                # time data
                self._t_initial[j] = start * self.dt
                self._t_final[j] = end * self.dt

        end_timer1 = timer()
        print(
            f"Training, validation and test data runtime: {timedelta(seconds=end_timer1-start_timer1)}"
        )
