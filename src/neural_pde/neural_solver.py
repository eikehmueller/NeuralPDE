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
        # list of neighbours, including the index itself, done from spherical_patch_coverine
        self._neighbour_list = [
            [j] + beta
            for j, beta in enumerate(self.spherical_patch_covering.neighbour_list)
        ]

    def forward(self, x):
        """Carry out a number of forward-Euler steps for the latent variables on the dual mesh

        :arg inputs: tensor of shape (B,n_patch,d_{latent}+d_{ancillary}) or (B,n_patch,d_{latent}+d_{ancillary})
        """

        if x.dim() == 2:
            index = (
                torch.tensor(self._neighbour_list)
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

                # ---- stage 2 ---- apply interaction model to obtain tensor of shape
                #                   (B,n_patch,d_{lat}^{dynamic})
                fz = self.interaction_model(z)


                # ---- stage 3 ---- pad with zeros in last dimension to obtain a tensor dY of shape
                #                   (B,n_patch,d_{lat}^{dynamic}+d_{lat}^{ancillary})
                dx = torch.nn.functional.pad( 
                    fz, (0, x.shape[-1] - fz.shape[-1]), mode="constant", value=0
                )
                # the padding (0, x.shape[-1] - fz.shape[-1]) is a list of 2*length of the source (fz.shape) 
                # the value of each is the amount of zeros to be added to each dimension. They come in zeros,
                # i.e. (0,1,1,1) pads a zero a column of zeros at the outermost dimension, then pads zeros
                # in the innermost dimensions on either side.


                # ---- stage 4 ---- update Y = Y + dt*dY
                x += self.stepsize * dx

            return x
        else:
            index = (
                torch.tensor(self._neighbour_list)
                .unsqueeze(0) 
                .unsqueeze(-1) 
                .expand((x.shape[0], -1, -1, x.shape[-1])) 
            )
            for _ in range(self.nsteps):
                # ---- stage 1 ---- gather to tensor Z of shape
                #                   (B,n_patch,4,d_{lat}^{dynamic}+d_{lat}^{ancillary})

                z = torch.gather(# gathers values along a specific axis. Way to extract values from a tensor
                    x.unsqueeze(-2).repeat((x.shape[0], x.shape[1], 4, x.shape[-1])),
                    1, 
                    index, 
                )

                # ---- stage 2 ---- apply interaction model to obtain tensor of shape
                #                   (B,n_patch,d_{lat}^{dynamic})
                fz = self.interaction_model(z)


                # ---- stage 3 ---- pad with zeros in last dimension to obtain a tensor dY of shape
                #                   (B,n_patch,d_{lat}^{dynamic}+d_{lat}^{ancillary})
                dx = torch.nn.functional.pad( 
                    fz, (0, x.shape[-1] - fz.shape[-1]), mode="constant", value=0
                )

                # ---- stage 4 ---- update Y = Y + dt*dY
                x += self.stepsize * dx

            return x

class Katies_NeuralSolver(torch.nn.Module):
    """This is Katie's attempt at replicating the Neural solver. This is to see if
    they do the same thing - I will try to do this without referencing Eike's one.

    This is a learnable function that that goes from
    R^{4, d_{lat}} -> R^{d_{lat}^{dyn}}

    Things that we take as arguments are
    
    spherical_patch_covering - this is where the red patches on the VOM are defined on the dual mesh
    interaction_model - the structure of the neural network that goes here 
                        this is of shape (4 * latent_dimension) (because we want 4 nodes )
                        output is of shape latent_dynamic_dim ???
    nsteps=1,  number of steps in the neural solver
    stepsize=1

    The previous layer of the neural network spits out a tensor of shape

    (B,n_{patches},d_{lat})

    which will be our input shape, and the next layer of the neural netowrk takes a tensor of shape

    (B,n_{patches},d_{lat}).

    So theoretically there should not really be any tensors changing shape.

    Also, the neighbour list is an ordered list, where,
    node[0] has neighbours neighbour_list[0]

    """

    # start by initializing the class
    def __init__(self, spherical_patch_covering, interaction_model, nsteps=1, stepsize=1):
        super().__init__()
        self.spherical_patch_covering = spherical_patch_covering
        self.interaction_model = interaction_model
        self.nsteps = nsteps
        self.stepsize = stepsize

        # neighbours list here
        self.neighbour_list = self.spherical_patch_covering.neighbour_list

    # define the forward method for this class
    # Given an input z_old, what is returned
    def forward(self, z_old):
        if z_old.dim() == 2:
            for _ in range(self.nsteps):
                # First, z_old is of shape (B, n_patch, d_lat)
                #print(f'The shape of z_old is {z_old.shape}')
                #print(self.neighbour_list)
                # but we want something that has shape (B, n_patch, d_lat, 4) 
                # first element of each 4 is the one in the middle
                z_unsqueezed = z_old.unsqueeze(2)
                #print(f'The shape of z_old is {z_unsqueezed.shape}')
                z_mid = z_unsqueezed.expand(-1, -1, 4)
                #print(f'The shape of z_old is now {z_mid.shape}')
                # now I want to access the elements in the tensors
                for n_patch in range(z_mid.shape[1]):
                    for d_lat in range(z_mid.shape[2]):
                        for beta in range(3):
                            # i is the same because they are from the same batch size
                            # The question is: what is the value of a neighbour of z_mid?
                            # What is the position of a neighbour of z_mid?
                            # if z_mid has position n_patch, it has position neighbours_list[patch][betha]
                            z_mid[n_patch][d_lat][beta + 1] = z_old[self.neighbour_list[n_patch][beta]][d_lat]
            
                F_theta = self.interaction_model(z_mid)
                # now we need to be careful because we are only updating the latent space variables
                #print(f'The shape of F_theta is {F_theta.shape}')
                num_dyn = F_theta.shape[1]
                z_old[:, :num_dyn] += self.stepsize * F_theta
        else:
            for _ in range(self.nsteps):
                # First, z_old is of shape (B, n_patch, d_lat)
                #print(f'The shape of z_old is {z_old.shape}')
                #print(self.neighbour_list)
                # but we want something that has shape (B, n_patch, d_lat, 4) 
                # first element of each 4 is the one in the middle
                z_unsqueezed = z_old.unsqueeze(3)
                #print(f'The shape of z_old is {z_unsqueezed.shape}')
                z_mid = z_unsqueezed.expand(-1, -1, -1, 4)
                #print(f'The shape of z_old is now {z_mid.shape}')
                # now I want to access the elements in the tensors
                for batch in range(z_mid.shape[0]):
                    for n_patch in range(z_mid.shape[1]):
                        for d_lat in range(z_mid.shape[2]):
                            for beta in range(3):
                                # i is the same because they are from the same batch size
                                # The question is: what is the value of a neighbour of z_mid?
                                # What is the position of a neighbour of z_mid?
                                # if z_mid has position n_patch, it has position neighbours_list[patch][betha]
                                z_mid[batch][n_patch][d_lat][beta + 1] = z_old[batch][self.neighbour_list[n_patch][beta]][d_lat]
            
                F_theta = self.interaction_model(z_mid)
                # now we need to be careful because we are only updating the latent space variables
                #print(f'The shape of F_theta is {F_theta.shape}')
                num_dyn = F_theta.shape[2]
                z_old[:, :, :num_dyn] += self.stepsize * F_theta

        return z_old