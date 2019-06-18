#!/usr/bin/env bash

floyd run --gpu --data gpostelnicu/datasets/ucf101-jpegs:/ucf101_jpegs --mode job --env pytorch-1.0 --message "finetune 29" "bash ops/floyd/train_finetune_layer4.sh"

