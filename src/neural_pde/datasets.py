import torch
import numpy as np
from scipy.stats import truncnorm
import json
import h5py
import tqdm
from torch.utils.data import Dataset
import itertools
import neural_pde.diagnostics as dg
from firedrake import (
    FunctionSpace,
    Function,
    SpatialCoordinate,
    UnitIcosahedralSphereMesh,
)
from firedrake import *
from timeit import default_timer as timer
from datetime import timedelta
#import sys
#sys.path.append("/home/katie795/software/gusto/gusto")
from gusto import (lonlatr_from_xyz, ShallowWaterParameters, ShallowWaterEquations, Domain,
    OutputParameters, IO, SSPRK3, DGUpwind, SemiImplicitQuasiNewton, RelativeVorticity, Divergence)

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
        t_initial = f["base/t_initial"]
        t_elapsed = f["base/t_elapsed"]
        metadata = json.loads(f["base/metadata"][()])
        dataset = SphericalFunctionSpaceDataset(
            int(f.attrs["n_func_in_dynamic"]),
            int(f.attrs["n_func_in_ancillary"]),
            int(f.attrs["n_func_target"]),
            int(f.attrs["n_ref"]),
            int(f.attrs["n_samples"]),
            data=np.asarray(data),
            t_initial=np.asarray(t_initial),
            t_elapsed=np.asarray(t_elapsed),
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
        t_initial=None,
        t_elapsed=None,
        mean=None,
        std=None,
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
        self.mesh = UnitIcosahedralSphereMesh(refinement_level=n_ref, name="IcosahedralMesh")  # create the mesh
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
            np.empty(self.n_samples, dtype=np.float64) if t_initial is None else t_initial
        ) 
        self._t_elapsed = (
            np.empty(self.n_samples, dtype=np.float64) if t_elapsed is None else t_elapsed
        )
        self.mean = (
            np.empty(self.n_func_in_dynamic
                    + self.n_func_in_ancillary
                    + self.n_func_target, dtype=np.float32) if mean is None else mean
        )
        self.std = (
            np.empty(self.n_func_in_dynamic
                    + self.n_func_in_ancillary
                    + self.n_func_target, dtype=np.float32) if std is None else std
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
            self._t_elapsed[idx],
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
            group.create_dataset("t_elapsed", data=self._t_elapsed)
            group.create_dataset("mean", data=self.mean)
            group.create_dataset("std", data=self.std)
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
        n_func_in_dynamic = 3
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
        self.nt = 1


    def generate(self):
        """Generate the data"""
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
            self._t_elapsed[j] = t_final

        self.mean = np.mean(self._data[:, 0, :], axis=1)
        self.std = np.std(self._data[:, 0, :], axis=1)
        return



class Projector:

    def __init__(self, W, V):
        """Class for projecting functions in from a 3D vector space to 3 separate scalar functions in V

        :arg W: function space W
        :arg V: function space V
        """
        self._W = W # this should be the BDM space!! or a DG space
        self._V = V # CG1 space
        self.phi = TestFunction(self._V)
        self.psi = TrialFunction(self._V)
        self._w_hdiv = Function(self._W)
        self._u = [
            Function(self._V),
            Function(self._V),
            Function(self._V),
        ]
        self._v = Function(self._V)
        self.a_mass = self.phi * self.psi * dx


    def create_linear_solver(self):

        if self._W.value_size == 3:
            lvs = []
            for j in range(3):
                n_hat = [0, 0, 0]
                n_hat[j] = 1
                b_hdiv = self.phi * inner(self._w_hdiv, as_vector(n_hat)) * dx
                lvp = LinearVariationalProblem(self.a_mass, b_hdiv, self._u[j])
                lvs.append(LinearVariationalSolver(lvp))
        elif self._W.value_size == 1:
            b_hscalar = self.phi * self._w_hdiv * dx
            lvp = LinearVariationalProblem(self.a_mass, b_hscalar, self._v)
            lvs = LinearVariationalSolver(lvp)
        else:
            print('Projector error: function space value is neither 1 nor 3.')

        return lvs

    def apply(self, w, u):
        """Project a specific function

        :arg w: function in W to project
        :arg u: list of three functions which will contain the result
        """
        lvs = self.create_linear_solver()

        if w.function_space().value_size == 3:
            self._w_hdiv.assign(w)
            for j in range(3):
                lvs[j].solve()
                u[j].assign(self._u[j])
        elif w.function_space().value_size == 1:
            self._w_hdiv.assign(w)
            lvs.solve()
            u.assign(self._v)
        else:
            print('Projector error: function space value is neither 1 nor 3.')

class ShallowWaterEquationsDataset(SphericalFunctionSpaceDataset):
    """Data set for Shallow Water Equations

    The input conists of the function fields (u,x,y,z) which represent a
    scalar function u and the three coordinate fields. The output is
    the same function, but rotated by some angle phi

    """

    def __init__(self, n_ref, nsamples, dt, t_final_max=10.0, omega=7.292e-5, g=9.8, t_interval=10, t_sigma=1, t_lowest=0, t_highest=10, save_diagnostics=False):
        """Initialise new instance

        :arg nref: number of mesh refinements
        :arg nsamples: number of samples
        :arg nt: number of timesteps
        :arg t_final_max: maximum final time
        :arg omega: rotation speed
        :arg g: gravitational acceleration 
        """
        self.n_func_in_dynamic = 3   # fixed for swes
        self.n_func_in_ancillary = 3 # x y and z coordinates
        self.n_func_target = 3       # fixed for swes

        self.omega = omega
        self.g = g
        self.save_diagnostics = save_diagnostics

        # initialise with the SphericalFunctionSpaceData
        super().__init__(
            self.n_func_in_dynamic, self.n_func_in_ancillary, self.n_func_target, n_ref, nsamples
        )

        self.metadata = {
            "omega": f"{self.omega:}",
            "t_final_max": f"{t_final_max:}",
            "t_lowest": f"{t_lowest}",
            "t_highest": f"{t_highest}",
        }
    
        self.dt = dt # number of timesteps
        self.t_final_max = t_final_max # final time
        self.t_interval = t_interval # the expected
        self.t_lowest = t_lowest
        self.t_highest = t_highest
        self.t_sigma = t_sigma

        x, y, z = SpatialCoordinate(self._fs.mesh()) # spatial coordinate
        self._x = Function(self._fs).interpolate(x) # collect data on x,y,z coordinates
        self._y = Function(self._fs).interpolate(y)
        self._z = Function(self._fs).interpolate(z)

    def generate_full_dataset(self):
        '''
        Generate the full data for the shallow water equations. The dataset used to train the model
        will be sampled from this data. This may take a while to load.
        '''

        start_timer = timer()

        L0 = 5960 # charactersistic length scale (mean height of water)
        R = 6371220 / L0 # radius of earth divided by length scale
        T0 = 12 * 24 * 60 * 60 / (2 * pi * R) # characteristic time scale - scales umax to 1

        element_order = 1 # CG method

        # BDM means Brezzi-Douglas-Marini finite element basis function
        domain = Domain(self.mesh, self.dt, 'BDM', element_order)
        print(domain.spaces("H1"))

        # ShallowWaterParameters are the physical parameters for the shallow water equations
        mean_depth = 1           # this is the parameter we nondimensionalise around
        g0 = self.g * (T0**2) / L0    # nondimensionalised g (m/s^2)
        Omega0 = self.omega * T0  # nondimensionalised omega (s^-1)

        # initialise parameters object
        parameters = ShallowWaterParameters(self.mesh, H=mean_depth, g=g0, Omega=Omega0)

        # set up the finite element form
        xyz = SpatialCoordinate(self.mesh)
        lon, lat, _ = lonlatr_from_xyz(*xyz)  # latitide and longitude 
        eqns = ShallowWaterEquations(domain, parameters)

        # output options 
        output = OutputParameters(dirname="gusto_output", dump_vtus=True, dump_nc=True, dump_diagnostics=True, dumpfreq=1, checkpoint=True, chkptfreq=1, multichkpt=True) # these have been modified so we get no output

        # choose which fields to record over the simulation
        #diagnostic_fields = [MeridionalComponent('u'), ZonalComponent('u'), SteadyStateError('D')]
        diagnostic_fields = [Divergence('u'), RelativeVorticity()]
        io = IO(domain, output, diagnostic_fields=diagnostic_fields)

        # the methods to solve the equations
        transported_fields = [SSPRK3(domain, "u"), SSPRK3(domain, "D")]
        transport_methods  = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D")]
        stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields, transport_methods)

        # setting the initial conditions for velocity
        u0 = stepper.fields("u")
        day = 24*60*60 / T0 # day in seconds
        u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
        uexpr = as_vector((-u_max*cos(lat)*sin(lon), u_max*cos(lat)*cos(lon), 0))
        u0.project(uexpr)

        # setting up initial conditions for height
        D0 = stepper.fields("D")
        g = parameters.g
        H = parameters.H

        # adding mountain ranges - these could all be varied!
        lamda_c = -pi/2.  # longitudinal centre of mountain (rad)
        phi_c = pi/6.     # latitudinal centre of mountain (rad)

        R0 = pi/9.                  # radius of mountain (rad)
        mountain_height = 2000 / L0 # height of mountain (m)

        rsq1 = min_value(R0**2, (lon - lamda_c)**2 + (lat - phi_c)**2)

        r1 = sqrt(rsq1)

        tpexpr = mountain_height *( (1 - r1/R0) )
        Dexpr = H - ((R * Omega0 * u_max + 0.5*u_max**2)*(sin(lat))**2) / g + tpexpr

        D0.interpolate(Dexpr)

        # reference velocity is zero, reference depth is H
        Dbar = Function(D0.function_space()).assign(H)
        stepper.set_reference_profiles([('D', Dbar)])

        # run the simulation
        stepper.run(t=0, tmax=self.t_final_max)
        end_timer = timer()
        print(f"Gusto runtime: {timedelta(seconds=end_timer-start_timer)}")
        return
    
    def prepare_for_model(self):
        '''
        Sample the data from the generated dataset and save as a np array
        '''
        start_timer1 = timer()

        with CheckpointFile('/home/katie795/NeuralPDE_workspace/scripts/results/gusto_output/chkpt.h5', 'r') as afile:
            print('we have opened the checkpoint file')
            mesh_h5 = afile.load_mesh("IcosahedralMesh")
            V_BDM = FunctionSpace(mesh_h5, "BDM", 2)
            V_DG = FunctionSpace(mesh_h5, "DG", 1)
            V_CG = FunctionSpace(mesh_h5, "CG", 1)

            h_inp = Function(V_CG) # input function for h
            h_tar = Function(V_CG) # target function for h

            p1 = Projector(V_BDM, V_CG)
            p2 = Projector(V_DG, V_CG)

            nt = int(self.t_final_max / self.dt)           

            diagnostics = dg.Diagnostics(V_BDM, V_CG)
            if self.save_diagnostics:
                file = VTKFile("results/gusto_output/diagnostics.pvd")
                
                for i in range(nt):
                    u_func = [Function(V_CG) for _ in range(3)]
                    u = afile.load_function(mesh_h5, "u", idx=i)
                    p1.apply(u, u_func)    # input u data

                    vorticity = diagnostics.vorticity(u)
                    divergence = diagnostics.divergence(u)
                    vorticity.rename("vorticity")
                    divergence.rename("divergence")
                    file.write(vorticity, divergence, u, t=i)
            
            interval = self.t_interval / self.dt
            sigma    = self.t_sigma    / self.dt
            
            for j in tqdm.tqdm(range(self.n_samples)):
                
                # randomly sample the generated data
                highest  = self.t_highest  / self.dt
                lowest   = self.t_lowest   / self.dt + 1 # there is an error in gusto at t = 0 in the divergence
                start = np.random.randint(lowest, highest)

                mu =  start + interval
                #t_norm = truncnorm((start - mu) / sigma, (highest - mu) / sigma, loc=mu, scale=sigma)
                end = start #+ interval #round(t_norm.rvs(1)[0]) CHANGE THIS BACK

                if j == 100:
                    print(f'Start timestep is {start}')
                    print(f'End timestep is {end}')
                
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
                self._data[j, 0, :] = h_inp.dat.data # h data
                self._data[j, 1, :] = divergence_inp.dat.data
                self._data[j, 2, :] = vorticity_inp.dat.data
                # coordinate data - auxiliary variables
                self._data[j, 3, :] = self._x.dat.data # x coord data
                self._data[j, 4, :] = self._y.dat.data # y coord data
                self._data[j, 5, :] = self._z.dat.data # z coord data
                # output data - target data
                self._data[j, 6, :] = h_tar.dat.data # h data
                self._data[j, 7, :] = divergence_tar.dat.data
                self._data[j, 8, :] = vorticity_tar.dat.data
                # time data
                self._t_initial[j]  = start * self.dt
                self._t_elapsed[j]  = (end - start) * self.dt
        
        self.mean = np.mean(self._data, axis=(0,2))
        self.std  = np.std(self._data, axis=(0,2))
        print(f'Mean is {self.mean}')
        print(f'Std is {self.std}')

        end_timer1 = timer()
        print(f"Training, validation and test data runtime: {timedelta(seconds=end_timer1-start_timer1)}")
        return