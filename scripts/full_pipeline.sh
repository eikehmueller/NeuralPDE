#!/bin/bash

set -e

bash generate_data.sh

python train.py

# evaluate the data
python evaluate.py \
    --data="../data/data_test_nref3_0.h5" 

python plot_data.py \
    --move_to_windows