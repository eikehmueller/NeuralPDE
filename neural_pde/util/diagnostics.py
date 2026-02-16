from firedrake import *


class Diagnostics:
    """Compute vorticity and divergence from HDiv velocity field on the sphere"""

    def __init__(self, fs_hdiv, fs, radius=1):
        """Initialise new instance

        :arg fs_hdiv: HDiv conforming space which contains the velocity
        :arg fs: function space to store vorticity and divergence in
        """
        self._fs = fs
        self._fs_hdiv = fs_hdiv
        mesh = fs_hdiv.mesh()
        X = SpatialCoordinate(mesh) / radius
        n = X 
        self._u = Function(fs_hdiv)
        phi = TestFunction(fs)
        psi = TrialFunction(fs)
        self._vorticity = Function(fs)
        self._divergence = Function(fs)
        solver_parameters = {
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "ksp_rtol": 1e-12,
        }
        a_mass = phi * psi * dx
        b_vorticity = (- inner(cross(n, grad(phi)), self._u)) * dx

        lvp_vorticity = LinearVariationalProblem(a_mass, b_vorticity, self._vorticity)
        self._lvs_vorticity = LinearVariationalSolver(
            lvp_vorticity, solver_parameters=solver_parameters
        )
        b_divergence = phi * div(self._u) * dx 
        lvp_divergence = LinearVariationalProblem(
            a_mass, b_divergence, self._divergence
        )
        self._lvs_divergence = LinearVariationalSolver(
            lvp_divergence, solver_parameters=solver_parameters
        )

    def vorticity(self, u):
        """Compute vorticity from given velocity

        :arg u: velocity function in HDiv space
        """
        self._u.assign(u)
        self._lvs_vorticity.solve()
        return Function(self._fs).assign(self._vorticity)

    def divergence(self, u):
        """Compute divergence from given velocity

        :arg u: velocity function in HDiv space
        """
        self._u.assign(u)
        self._lvs_divergence.solve()
        return Function(self._fs).assign(self._divergence)

if __name__=='__main__':
    nref = 5

    mesh = UnitIcosahedralSphereMesh(nref)
    fs = FunctionSpace(mesh, "CG", 1)
    fs_hdiv = FunctionSpace(mesh, "BDM", 1)
    X = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(X)
    x, y, z = X

    u1 = Function(fs_hdiv, name="velocity").project(
        as_vector([-y/2, x/2, 0])
    )

    diagnostics = Diagnostics(fs_hdiv, fs)

    vorticity = diagnostics.vorticity(u1)
    divergence = diagnostics.divergence(u1)
    vorticity.rename("vorticity")
    divergence.rename("divergence")

    file = VTKFile("diagnostics.pvd")
    file.write(vorticity, divergence, u1)