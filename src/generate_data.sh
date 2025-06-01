#!/bin/bash
NREF=5
DEGREE=4
OMEGA=1.0
TFINALMAX=1.57079632679 # = pi/2

python data_generator.py \
    --filename data/data_train_nref${NREF}.h5 \
    --nref ${NREF} \
    --degree ${DEGREE} \
    --omega ${OMEGA} \
    --tfinalmax ${TFINALMAX} \
    --nsamples 1024 \
    --seed 152167

python data_generator.py \
    --filename data/data_valid_nref${NREF}.h5 \
    --nref ${NREF} \
    --degree ${DEGREE} \
    --omega ${OMEGA} \
    --tfinalmax ${TFINALMAX} \
    --nsamples 128 \
    --seed 251373

python data_generator.py \
    --filename data/data_test_nref${NREF}.h5 \
    --nref ${NREF} \
    --degree ${DEGREE} \
    --omega ${OMEGA} \
    --tfinalmax ${TFINALMAX} \
    --nsamples 128 \
    --seed 515157