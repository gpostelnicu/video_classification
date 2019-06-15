#!/usr/bin/env bash

PYTHONPATH=`pwd` python bin/train.py \
    --base_dir /ucf101_jpegs/ \
    --train_list resources/train_dat.txt \
    --test_list resources/test_dat.txt \
    --config resources/tiny_config.yaml \
    --save_file /output/tiny.pt \
    --num_epochs 20 \
    --num_workers 2 \
    --print_every_n 10 \
    --num_frames 29
