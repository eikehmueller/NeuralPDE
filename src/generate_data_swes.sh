#!/bin/bash
mkdir -p data/

NREF=5
OMEGA=1e-4
TFINALMAX=0.01
G=9.8
NT=100
TINTERVAL=50
TSIGMA=25

python data_generator_swes.py \
    --filename data/data_train_swes_nref_${NREF}_${TFINALMAX}.h5 \
    --nref ${NREF} \
    --omega ${OMEGA} \
    --g ${G} \
    --nt ${NT} \
    --tfinalmax ${TFINALMAX} \
    --nsamples 128 \
    --t_interval ${TINTERVAL} \
    --t_sigma ${TSIGMA} \
    --t_lowest 0 \
    --t_highest 800 \
    --regenerate_data \

python data_generator_swes.py \
    --filename data/data_valid_swes_nref_${NREF}_${TFINALMAX}.h5 \
    --nref ${NREF} \
    --omega ${OMEGA} \
    --g ${G} \
    --nt ${NT} \
    --tfinalmax ${TFINALMAX} \
    --nsamples 128 \
    --t_interval ${TINTERVAL} \
    --t_sigma ${TSIGMA} \
    --t_lowest 0 \
    --t_highest 800 \

python data_generator_swes.py \
    --filename data/data_test_swes_nref_${NREF}_${TFINALMAX}.h5 \
    --nref ${NREF} \
    --omega ${OMEGA} \
    --g ${G} \
    --nt ${NT} \
    --tfinalmax ${TFINALMAX} \
    --nsamples 128 \
    --t_interval ${TINTERVAL} \
    --t_sigma ${TSIGMA} \
    --t_lowest 800 \
    --t_highest ${NT} \