"""Patch encoder. Encodes the field information on each patch into latent space"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf


class PatchEncoder(tf.keras.layers.Layer):
    """Collective encoding to latent space

    Apply transformation to tensor of shape

        (B,n_{func},n_{dof per patch})

    to obtain tensor of shape

        (B,d_{latent}+d_{ancillary})

    here n_{func} = n_{dynamic} + n_{ancillary} is the total number of fields,
    consisting of both dynamic and ancillary fields. The mapping has a
    block-structure in the sense that

        [*,:n_dynamic,:] gets mapped to [*,:n_latent]

    and

        [*,n_dynamic:,:] gets mapped to [*,n_latent:].

    """

    def __init__(self, n_dynamic, latent_dim, ancillary_dim):
        """Initialise instance

        :arg n_dynamic: number of dynamic fields
        :arg latent_dim: dimension n_{latent} of latent space
        :arg ancillary_dim: dimension n_{ancil} of latent ancillary space"""
        super().__init__()
        self.n_dynamic = n_dynamic
        self.latent_dim = latent_dim
        self.ancillary_dim = ancillary_dim

    def build(self, input_shape):
        """Construct weights

        :arg input_shape: shape of the input tensor
        """
        self.W_dynamic = self.add_weight(
            shape=(
                self.n_dynamic,
                input_shape[-1],
                self.latent_dim,
            ),
            initializer="random_normal",
            trainable=True,
        )
        self.b_dynamic = self.add_weight(
            shape=(self.latent_dim,),
            initializer="zeros",
            trainable=True,
        )
        self.W_ancillary = self.add_weight(
            shape=(
                input_shape[-2] - self.n_dynamic,
                input_shape[-1],
                self.ancillary_dim,
            ),
            initializer="random_normal",
            trainable=True,
        )
        self.b_ancillary = self.add_weight(
            shape=(self.ancillary_dim,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        """Call layer and apply linear transformation

        Returns a tensor of shape

            (B,n_{patches},d_{latent}+d_{ancillary})

        where B is the batch size

        :arg inputs: a tensor of shape (B,n_{patches},n_{func},n_{points per patch})
        """
        return tf.concat(
            [
                tf.einsum(
                    f"bmij,ijk->bmk",
                    inputs[..., : self.n_dynamic, :],
                    self.W_dynamic,
                )
                + self.b_dynamic,
                tf.einsum(
                    f"bmij,ijk->bmk",
                    inputs[..., self.n_dynamic :, :],
                    self.W_ancillary,
                )
                + self.b_ancillary,
            ],
            axis=-1,
        )
