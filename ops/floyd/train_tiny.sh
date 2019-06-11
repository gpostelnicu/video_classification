#!/usr/bin/env bash

PYTHONPATH=`pwd` python bin/train.py \
    --base_dir /ucf101_jpegs/ \
    --train_list resources/train_dat.txt \
    --test_list resources/test_dat.txt \
    --config resources/tiny_config.yaml \
    --save_prefix /output/tiny \
    --num_epochs 20 \
    --num_workers 0 \
    --print_every_n 10 \
    --num_frames 50
