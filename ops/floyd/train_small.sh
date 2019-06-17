#!/usr/bin/env bash

PYTHONPATH=`pwd` python bin/train.py \
    --base_dir /ucf101_jpegs/ \
    --train_list resources/train_dat.txt \
    --test_list resources/test_dat.txt \
    --config resources/small_config.yaml \
    --save_file /output/small.pth \
    --num_epochs 10 \
    --num_workers 4 \
    --print_every_n 10 \
    --num_frames 29
