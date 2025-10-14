#!/bin/bash
mkdir -p data/

NREF=4
OMEGA=7.292e-5
TFINALMAX=0.0001 # ~ pi
G=9.8
NT=10

python data_generator_swes.py \
    --filename data/data_train_swes_nref_${NREF}_${TFINALMAX}.h5 \
    --nref ${NREF} \
    --omega ${OMEGA} \
    --g ${G} \
    --nt ${NT} \
    --tfinalmax ${TFINALMAX} \
    --nsamples 4096 \

python data_generator_swes.py \
    --filename data/data_valid_swes_nref_${NREF}_${TFINALMAX}.h5 \
    --nref ${NREF} \
    --omega ${OMEGA} \
    --g ${G} \
    --nt ${NT} \
    --tfinalmax ${TFINALMAX} \
    --nsamples 32 \

python data_generator_swes.py \
    --filename data/data_test_swes_nref_${NREF}_${TFINALMAX}.h5 \
    --nref ${NREF} \
    --omega ${OMEGA} \
    --g ${G} \
    --nt ${NT} \
    --tfinalmax ${TFINALMAX} \
    --nsamples 32 \