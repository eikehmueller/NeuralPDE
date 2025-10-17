#!/bin/bash
mkdir -p data/

NREF=3
OMEGA=7.292e-5
TFINALMAX=0.0001
G=9.8
NT=5
TINTERVAL=2
TSIGMA=1

python data_generator_swes.py \
    --filename data/data_train_swes_nref_${NREF}_${TFINALMAX}.h5 \
    --nref ${NREF} \
    --omega ${OMEGA} \
    --g ${G} \
    --nt ${NT} \
    --tfinalmax ${TFINALMAX} \
    --nsamples 32 \
    --t_interval ${TINTERVAL} \
    --t_sigma ${TSIGMA} \
    --regenerate_data \

python data_generator_swes.py \
    --filename data/data_valid_swes_nref_${NREF}_${TFINALMAX}.h5 \
    --nref ${NREF} \
    --omega ${OMEGA} \
    --g ${G} \
    --nt ${NT} \
    --tfinalmax ${TFINALMAX} \
    --nsamples 16 \
    --t_interval ${TINTERVAL} \
    --t_sigma ${TSIGMA} \

python data_generator_swes.py \
    --filename data/data_test_swes_nref_${NREF}_${TFINALMAX}.h5 \
    --nref ${NREF} \
    --omega ${OMEGA} \
    --g ${G} \
    --nt ${NT} \
    --tfinalmax ${TFINALMAX} \
    --nsamples 16 \
    --t_interval ${TINTERVAL} \
    --t_sigma ${TSIGMA} \