#!/bin/bash
mkdir -p data/

NREF=3
OMEGA=1e-4
TFINALMAX=50
G=9.8
NT=5000
TINTERVAL=20
TSIGMA=10
TSPLIT=4500

python data_generator_swes.py \
    --filename data/data_train_swes_nref_${NREF}_${TFINALMAX}.h5 \
    --nref ${NREF} \
    --omega ${OMEGA} \
    --g ${G} \
    --nt ${NT} \
    --tfinalmax ${TFINALMAX} \
    --nsamples 4096 \
    --t_interval ${TINTERVAL} \
    --t_sigma ${TSIGMA} \
    --t_lowest 0 \
    --t_highest ${TSPLIT} \
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
    --t_highest ${TSPLIT} \

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
    --t_lowest ${TSPLIT} \
    --t_highest ${NT} \