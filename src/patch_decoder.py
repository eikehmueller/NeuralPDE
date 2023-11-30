"""Patch encoder. Decodes the field information back from latent space"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf


class PatchDecoder(tf.keras.layers.Layer):
    """Collective decoding from latent space

    Apply transformation to tensor of shape

        (B,n_{patches},d_{latent})

    to obtain tensor of shape

        (B,n_{patches},n_{func},n_{dof per patch})
    """

    def __init__(self, n_func, patch_size):
        """Initialise instance

        :arg n_func: number of functions in output
        :arg patch_size: number of dofs per patch n_{dofs per patch})
        """
        super().__init__()
        self.n_func = n_func
        self.patch_size = patch_size

    def build(self, input_shape):
        """Construct weights

        :arg input_shape: shape of the input tensor
        """
        self.W = self.add_weight(
            shape=(input_shape[-1], self.n_func, self.patch_size),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.n_func, self.patch_size),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        """Call layer and apply linear transformation

        Returns a tensor of shape

            (B,n_{patches},n_{func},n_{points per patch})

        where B is the batch size.

        :arg inputs: a tensor of shape (B,n_{patches},d_{latent})
        """
        input_dim = len(inputs.shape)
        return tf.einsum(f"bmk,kij->bmij", inputs, self.W) + self.b
