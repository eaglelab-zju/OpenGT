#!/usr/bin/env bash

MODEL=NodeFormer
CONFIG=webkb-cor-NodeFormer

python main.py --cfg configs/${MODEL}/${CONFIG}.yaml

tensorboard --logdir=results/${CONFIG} --port=6006
