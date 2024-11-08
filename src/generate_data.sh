#!/bin/bash
NREF=3
DEGREE=4
PHI=0.5

python data_generator.py \
    --filename data/data_train.h5 \
    --nref ${NREF} \
    --degree ${DEGREE} \
    --phi ${PHI} \
    --nsamples 512 \
    --seed 152167

python data_generator.py \
    --filename data/data_valid.h5 \
    --nref ${NREF} \
    --degree ${DEGREE} \
    --phi ${PHI} \
    --nsamples 64 \
    --seed 251373

python data_generator.py \
    --filename data/data_test.h5 \
    --nref ${NREF} \
    --degree ${DEGREE} \
    --phi ${PHI} \
    --nsamples 64 \
    --seed 515157