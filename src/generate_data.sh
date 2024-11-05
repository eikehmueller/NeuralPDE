#!/bin/bash
NREF=1
DEGREE=1
PHI=0.1963

python data_generator.py \
    --filename data/data_train.h5 \
    --nref ${NREF} \
    --degree ${DEGREE} \
    --phi ${PHI} \
    --nsamples 128 \
    --seed 152167

python data_generator.py \
    --filename data/data_valid.h5 \
    --nref ${NREF} \
    --degree ${DEGREE} \
    --phi ${PHI} \
    --nsamples 32 \
    --seed 251373