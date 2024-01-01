import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from firedrake import (
    UnitIcosahedralSphereMesh,
    Function,
    FunctionSpace,
    SpatialCoordinate,
)

# NB: need to import tensorflow *after* firedrake ...
import tensorflow as tf

from neural_pde.spherical_patch_covering import SphericalPatchCovering
from neural_pde.patch_interpolation import (
    FunctionToPatchInterpolationLayer,
    PatchToFunctionInterpolationLayer,
)
from neural_pde.patch_encoder import PatchEncoder
from neural_pde.patch_decoder import PatchDecoder
from neural_pde.neural_solver import NeuralSolver
from neural_pde.data_generator import AdvectionDataGenerator


############################################################################
# M A I N (for testing)
############################################################################

if __name__ == "__main__":
    # construct spherical patch covering
    spherical_patch_covering = SphericalPatchCovering(0, 4)

    print(f"number of patches               = {spherical_patch_covering.n_patches}")
    print(f"patchsize                       = {spherical_patch_covering.patch_size}")
    print(f"number of points in all patches = {spherical_patch_covering.n_points}")

    mesh = UnitIcosahedralSphereMesh(1)
    V = FunctionSpace(mesh, "CG", 1)

    # number of dynamic fields: scalar tracer
    n_dynamic = 1
    # number of ancillary fields: x-, y- and z-coordinates
    n_ancillary = 3
    # dimension of latent space
    latent_dim = 17
    # dimension of ancillary space
    ancillary_dim = 6
    # number of output fields: scalar tracer
    n_output = 1

    # encoder models
    # dynamic encoder model: map all fields to the latent space
    dynamic_encoder_model = tf.keras.Sequential(
        [
            tf.keras.Input(
                shape=(n_dynamic + n_ancillary, spherical_patch_covering.patch_size)
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=latent_dim),
        ]
    )
    # ancillary encoder model: map ancillary fields to ancillary space
    ancillary_encoder_model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(n_ancillary, spherical_patch_covering.patch_size)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=ancillary_dim),
        ]
    )

    # decoder model: map from latent and ancillary space to output fields
    decoder_model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(latent_dim + ancillary_dim,)),
            tf.keras.layers.Dense(units=n_output * spherical_patch_covering.patch_size),
            tf.keras.layers.Reshape(
                target_shape=(n_output, spherical_patch_covering.patch_size)
            ),
        ]
    )

    # interaction model
    interaction_model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(3, latent_dim + ancillary_dim)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=latent_dim),
        ]
    )

    # put everything together into one model that maps from the input to targets
    model = tf.keras.Sequential(
        [
            FunctionToPatchInterpolationLayer(V, spherical_patch_covering),
            PatchEncoder(dynamic_encoder_model, ancillary_encoder_model),
            NeuralSolver(
                spherical_patch_covering,
                interaction_model=interaction_model,
                latent_dim=latent_dim,
                nsteps=4,
                stepsize=0.1,
            ),
            PatchDecoder(decoder_model),
            PatchToFunctionInterpolationLayer(spherical_patch_covering, V),
        ]
    )

    model.compile(optimizer="adam", loss="mse", metrics=[])

    generator = AdvectionDataGenerator(V, 1.0)
    batch_size = 8
    dataset = (
        tf.data.Dataset.from_generator(
            generator, output_signature=generator.output_signature
        )
        .take(64)
        .batch(batch_size, drop_remainder=True)
    )

    log_dir = "./tb_logs/"

    tboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, profile_batch="0,1"
    )

    model.fit(dataset, epochs=10, callbacks=[tboard_callback])
