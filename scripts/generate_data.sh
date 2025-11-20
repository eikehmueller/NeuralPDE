#!/bin/bash
mkdir -p ../data/

NREF=3
DEGREE=4
OMEGA=1.0
TFINALMAX=0 # ~ pi

python data_generator.py \
    --filename ../data/data_train_nref${NREF}_${TFINALMAX}.h5 \
    --nref ${NREF} \
    --degree ${DEGREE} \
    --omega ${OMEGA} \
    --tfinalmax ${TFINALMAX} \
    --nsamples 4096 \
    --seed 152167

python data_generator.py \
    --filename ../data/data_valid_nref${NREF}_${TFINALMAX}.h5 \
    --nref ${NREF} \
    --degree ${DEGREE} \
    --omega ${OMEGA} \
    --tfinalmax ${TFINALMAX} \
    --nsamples 128 \
    --seed 251373

python data_generator.py \
    --filename ../data/data_test_nref${NREF}_${TFINALMAX}.h5 \
    --nref ${NREF} \
    --degree ${DEGREE} \
    --omega ${OMEGA} \
    --tfinalmax ${TFINALMAX} \
    --nsamples 128 \
    --seed 515157