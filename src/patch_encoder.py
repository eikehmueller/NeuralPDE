"""Patch encoder. Encodes the field information on each patch into latent space"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf


class PatchEncoder(tf.keras.layers.Layer):
    """Collective encoding to latent space

    Apply transformation to tensor of shape

        (*,n_{func},n_{dof per patch})

    to obtain tensor of shape

        (*,n_{func},n_{latent})
    """

    def __init__(self, latent_dim):
        """Initialise instance

        :arg latent_dim: dimension n_{latent} of latent space"""
        super().__init__()
        self.latent_dim = latent_dim

    def build(self, input_shape):
        """Construct weights

        :arg input_shape: shape of the input tensor
        """
        self.W = self.add_weight(
            shape=(*input_shape[-2:], self.latent_dim),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.latent_dim,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        """Call layer and apply linear transformation

        Returns a tensor of shape

            ([B],n_{patches},n_{latent})

        where B is the (optional) batch size

        :arg inputs: a tensor of shape ([B],n_{patches},n_{func},n_{points per patch})
        """
        input_dim = len(inputs.shape)
        extra_indices = "abcdefgh"[: input_dim - 2]
        return (
            tf.einsum(f"{extra_indices}ij,ijk->{extra_indices}k", inputs, self.W)
            + self.b
        )
