#!/bin/bash
mkdir -p ../data/

NREF=3
OMEGA=1e-4
TFINALMAX=1
G=9.8
DT=0.01

TINTERVAL=0.0
TSIGMA=0.0
TSPLIT=0.8

python data_generator_swes.py \
    --filename ../data/data_train_swes_nref${NREF}_tlength${TINTERVAL}_tfinalmax${TFINALMAX}.h5 \
    --nref ${NREF} \
    --omega ${OMEGA} \
    --g ${G} \
    --dt ${DT} \
    --tfinalmax ${TFINALMAX} \
    --nsamples 4096 \
    --t_interval ${TINTERVAL} \
    --t_sigma ${TSIGMA} \
    --t_lowest 0 \
    --t_highest ${TSPLIT} \

python data_generator_swes.py \
    --filename ../data/data_valid_swes_nref${NREF}_tlength${TINTERVAL}_tfinalmax${TFINALMAX}.h5 \
    --nref ${NREF} \
    --omega ${OMEGA} \
    --g ${G} \
    --dt ${DT} \
    --tfinalmax ${TFINALMAX} \
    --nsamples 128 \
    --t_interval ${TINTERVAL} \
    --t_sigma ${TSIGMA} \
    --t_lowest 0 \
    --t_highest ${TSPLIT} \

python data_generator_swes.py \
    --filename ../data/data_test_swes_nref${NREF}_tlength${TINTERVAL}_tfinalmax${TFINALMAX}.h5 \
    --nref ${NREF} \
    --omega ${OMEGA} \
    --g ${G} \
    --dt ${DT} \
    --tfinalmax ${TFINALMAX} \
    --nsamples 128 \
    --t_interval ${TINTERVAL} \
    --t_sigma ${TSIGMA} \
    --t_lowest ${TSPLIT} \
    --t_highest ${TFINALMAX} \