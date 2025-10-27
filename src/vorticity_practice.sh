#!/bin/bash
mkdir -p data/

NREF=4
OMEGA=1e-4
TFINALMAX=0.001
G=9.8
NT=10
TINTERVAL=10
TSIGMA=5

python data_generator_swes.py \
    --filename data/data_train_swes_animation_nref_${NREF}_${TFINALMAX}.h5 \
    --nref ${NREF} \
    --omega ${OMEGA} \
    --g ${G} \
    --nt ${NT} \
    --tfinalmax ${TFINALMAX} \
    --nsamples 1 \
    --t_interval ${TINTERVAL} \
    --t_sigma ${TSIGMA} \
    --t_lowest 0 \
    --t_highest 10 \
    --regenerate_data \