#!/bin/bash
mkdir -p data/

NREF=4
OMEGA=7.292e-5
TFINALMAX=0.001
G=9.8
NT=10000
TINTERVAL=50
TSIGMA=25

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