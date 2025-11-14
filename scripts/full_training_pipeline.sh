#!/bin/bash

set -e

cp config.toml config_records/config1

python train.py

# evaluate the data
python evaluate.py \
    --data="../data/data_test_nref3_0.h5" \
    --move_to_windows
