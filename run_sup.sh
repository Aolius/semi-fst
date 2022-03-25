#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python T5_Large_filter.py \
    -style 0 \
    -dataset em \
    -order em.sup \
    -batch_size 8 \
    -val_batch_size 16 \
    -lr 2e-5