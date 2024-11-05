#!/bin/bash
NREF=2
DEGREE=4
PHI=0.0

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
    --nsamples 32 \
    --seed 251373