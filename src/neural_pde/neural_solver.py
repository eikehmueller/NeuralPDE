import torch

class NeuralSolver(torch.nn.Module):
    """Neural solver which integrates forward the equations of motion in latent space.

    A state in latent space can be written as [Y^{(k)}_{i,j},a_{i,j}] where i is the index of
    the vertex on the dual mesh that defines the spherical patch covering.

        Y^{(k)}_{i,j} for j = 0,1,...,d_{latent}-1

    is the latent state vector which evolves and

        a_{i,j} for j = 0,1,...,d_{ancillary}-1

    is the ancillary state vector which remains unchanged.

    Assume that we have some model parametrised by learnable parameters theta with

        Phi_theta([Y_{i,j},a_{i,j}]) = Phi_theta(Y_{i_0,j'},Y_{i_1,j'},Y_{i_2,j'};
                                                 a_{i_0,j'},a_{i_1,j},a_{i_2,j'})

    where {i_0,i_1,i_2} is the set of vertices which contains the vertex i itself and its two
    neighbours. Note that the input shape of Phi_theta is (3,d_{latent}+d_{ancillary}) and the
    output shape is (d_{latent},).

    Then the update implemented here is the forward-Euler model

        Y^{(k+1)}_{i,j} = Y^{(k)}_{i,j} + dt * Phi_theta([Y^{(k)}_{i,j},a_{i,j}])

    which comes from

    dY/dt = Phi_theta

    and Phi_theta is a learnable function.
    """

    def __init__(
        self,
        spherical_patch_covering,
        interaction_model,
        nsteps=1,
        stepsize=1.0,
        assert_testing=None
    ):
        """Initialise new instance

        :arg spherical_patch_covering: the spherical patch covering that defines the topology and
            local patches
        :arg interaction_model: model that describes the interactions Phi_theta
        :arg nsteps: number of forward-Euler steps
        :arg stepsize: size dt of steps
        """
        super().__init__()
        self.spherical_patch_covering = spherical_patch_covering
        self.interaction_model = interaction_model
        self.nsteps = nsteps
        self.stepsize = stepsize
        self._neighbour_list = [
            [j] + beta
            for j, beta in enumerate(self.spherical_patch_covering.neighbour_list)
        ]
        self.assert_testing = assert_testing

    def forward(self, x):
        """Carry out a number of forward-Euler steps for the latent variables on the dual mesh

        :arg inputs: tensor of shape (B,n_patch,d_{latent}+d_{ancillary}) or (B,n_patch,d_{latent}+d_{ancillary})
        """
        device = x.device
        if x.dim() == 2 or x.shape[0] == 1:
            x = x.squeeze(0)
            index = (
                torch.tensor(self._neighbour_list, device=device)
                .unsqueeze(-1) 
                .expand((-1, -1, x.shape[-1])) 
                )
            
            for _ in range(self.nsteps):
                # ---- stage 1 ---- gather to tensor Z of shape
                #                   (n_patch,4,d_{lat}^{dynamic}+d_{lat}^{ancillary})
                z = torch.gather(
                    x.unsqueeze(-2).repeat((x.shape[0], 4, x.shape[-1])),
                    0, 
                    index, # must have same number of dimensions as the input
                )

                if self.assert_testing is not None:
                    assert z.shape[0] == self.assert_testing["n_patches"], "z[1] is not equal to n_patches"
                    assert z.shape[1] == 4,                                "z[2] is not equal to 4"
                    assert z.shape[2] == self.assert_testing["d_lat"],     "z[3] is not equal to d_lat"

                # ---- stage 2 ---- apply interaction model to obtain tensor of shape
                #                   (n_patch,d_{lat}^{dynamic})
                fz = self.interaction_model(z)

                if self.assert_testing is not None:
                    assert fz.shape[0] == self.assert_testing["n_patches"], "fz[0] is not equal to n_patches"
                    assert fz.shape[1] == self.assert_testing["d_dyn"],     "fz[1] is not equal to d_dyn"


                # ---- stage 3 ---- pad with zeros in last dimension to obtain a tensor dY of shape
                #                   (n_patch,d_{lat}^{dynamic}+d_{lat}^{ancillary})
                dx = torch.nn.functional.pad( 
                    fz, (0, x.shape[-1] - fz.shape[-1]), mode="constant", value=0
                )

                # ---- stage 4 ---- update Y = Y + dt*dY
                x += self.stepsize * dx

            return x
        else:
            index = (
                torch.tensor(self._neighbour_list, device=device)
                .unsqueeze(0) 
                .unsqueeze(-1) 
                .expand((x.shape[0], -1, -1, x.shape[-1])) 
            )
            for _ in range(self.nsteps):
                # ---- stage 1 ---- gather to tensor Z of shape
                #                   (B,n_patch,4,d_{lat}^{dynamic}+d_{lat}^{ancillary})

                z = torch.gather(
                    x.unsqueeze(-2).repeat((x.shape[0], x.shape[1], 4, x.shape[-1])),
                    1, 
                    index, 
                )

                if self.assert_testing is not None:
                    assert z.shape[0] == self.assert_testing["batchsize"], "z[0] is not equal to batch size"
                    assert z.shape[1] == self.assert_testing["n_patches"], "z[1] is not equal to n_patches"
                    assert z.shape[2] == 4,                                "z[2] is not equal to 4"
                    assert z.shape[3] == self.assert_testing["d_lat"],     "z[3] is not equal to d_lat"

                # ---- stage 2 ---- apply interaction model to obtain tensor of shape
                #                   (B,n_patch,d_{lat}^{dynamic})
                fz = self.interaction_model(z)

                if self.assert_testing is not None:
                    assert fz.shape[0] == self.assert_testing["batchsize"], "fz[0] is not equal to batch size"
                    assert fz.shape[1] == self.assert_testing["n_patches"], "fz[1] is not equal to n_patches"
                    assert fz.shape[2] == self.assert_testing["d_dyn"],     "fz[2] is not equal to d_dyn"

                # ---- stage 3 ---- pad with zeros in last dimension to obtain a tensor dY of shape
                #                   (B,n_patch,d_{lat}^{dynamic}+d_{lat}^{ancillary})
                dx = torch.nn.functional.pad( 
                    fz, (0, x.shape[-1] - fz.shape[-1]), mode="constant", value=0
                )

                # ---- stage 4 ---- update Y = Y + dt*dY
                x += self.stepsize * dx

            return x