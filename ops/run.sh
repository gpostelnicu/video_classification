#!/usr/bin/env bash

floyd run --gpu --data gpostelnicu/datasets/ucf101-jpegs:/ucf101_jpegs --mode job --env pytorch-1.0 --message "small 29" "bash ops/floyd/train_small.sh"

