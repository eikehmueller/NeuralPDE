"""Patch encoder. Encodes the field information on each patch into latent space"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf


class PatchEncoder(tf.keras.layers.Layer):
    """Collective encoding to latent space

    Apply transformation to tensor of shape

        (B,n_{patches},n_{func},n_{dof per patch})

    to obtain tensor of shape

        (B,n_{patches},d_{latent}+d_{ancillary})

    here n_{func} = n_{dynamic} + n_{ancillary} is the total number of fields,
    consisting of both dynamic and ancillary fields.

    The mapping has block-structure in the sense that

        [b,i,:,:] gets mapped to [b,i,:d_{latent}]

    by applying dynamic_encoder_model for each index pair (b,i)

    and

        [b,i,n_{dynamic}:,:] gets mapped to [p,i,d_{latent}:].

    by applying ancillary_encoder_model for each index pair (b,i).

    This means that the latent variables in the processor will depend both on the
    dynamic- and on the ancillary fields on the patches and the ancillary variables in the
    processor will only depend on the ancillary fields on the patches.
    """

    def __init__(self, dynamic_encoder_model, ancillary_encoder_model):
        """Initialise instance

        :arg dynamic_encoder model: maps tensors of shape (n_{dynamic}+n_{ancillary},patch_size) to
            tensors of shape (d_{latent},)
        :arg ancillary_encoder_model: maps tensors of shape (n_{ancillary},patch_size) to
            tensors of shape (d_{ancillary},)
        """
        super().__init__()
        self._dynamic_encoder_model = dynamic_encoder_model
        self._ancillary_encoder_model = ancillary_encoder_model
        self._n_latent = self._ancillary_encoder_model.layers[0].input_shape[-2]

    def call(self, inputs):
        """Call layer and apply linear transformation

        Returns a tensor of shape

            (B,n_{patches},d_{latent}+d_{ancillary})

        where B is the batch size

        :arg inputs: a tensor of shape (B,n_{patches},n_{func},n_{points per patch})
        """
        return tf.concat(
            [
                tf.stack(
                    [
                        self._dynamic_encoder_model(patch)
                        for patch in tf.unstack(inputs)
                    ],
                    axis=0,
                ),
                tf.stack(
                    [
                        self._ancillary_encoder_model(patch[:, -self._n_latent :])
                        for patch in tf.unstack(inputs)
                    ],
                    axis=0,
                ),
            ],
            axis=-1,
        )
