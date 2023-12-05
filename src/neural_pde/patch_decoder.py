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

    def __init__(self, decoder_model):
        """Initialise instance

        :arg decoder_model: model used for decoding
        """
        super().__init__()
        self._decoder_model = decoder_model

    def call(self, inputs):
        """Call layer and apply linear transformation

        Returns a tensor of shape

            (B,n_{patches},n_{func},n_{points per patch})

        where B is the batch size.

        :arg inputs: a tensor of shape (B,n_{patches},d_{latent}+{d_ancillary})
        """
        return tf.stack(
            [self._decoder_model(patch) for patch in tf.unstack(inputs)],
            axis=0,
        )
