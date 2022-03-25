#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python T5_Large_filter.py \
    -unsup \
    -style 0 \
    -ratio 1.0 \
    -dataset em \
    -order em-semi \
    -pre_step 2000 \
    -batch_size 8 \
    -unsup_batch_size 56 \
    -val_batch_size 16 \
    -lr 2e-5  \
    -aug_choice spell \
    -aug_p 0.1  \
    -filter cls \
    -phi 0.4  \
    -n_step 1000 \
    -sigma 0.8