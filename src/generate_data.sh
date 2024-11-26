#!/bin/bash
NREF=4
DEGREE=4
PHI=0.5

python data_generator.py \
    --filename data/data_train_nref4.h5 \
    --nref ${NREF} \
    --degree ${DEGREE} \
    --phi ${PHI} \
    --nsamples 1024 \
    --seed 152167

python data_generator.py \
    --filename data/data_valid_nref4.h5 \
    --nref ${NREF} \
    --degree ${DEGREE} \
    --phi ${PHI} \
    --nsamples 128 \
    --seed 251373

python data_generator.py \
    --filename data/data_test_nref4.h5 \
    --nref ${NREF} \
    --degree ${DEGREE} \
    --phi ${PHI} \
    --nsamples 128 \
    --seed 515157