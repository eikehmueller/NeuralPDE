
class Projector:

    def __init__(self, W, V):
        """Class for projecting functions in from a 3D vector space to 3 
        separate scalar functions in V

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
