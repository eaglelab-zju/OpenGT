#!/usr/bin/env bash

MODEL=GPS+GE
DATASET=wn-chameleon

python main.py --cfg configs/${MODEL}/${DATASET}-${MODEL}.yaml --repeat 3

tensorboard --logdir=results/${DATASET}-${MODEL} --port=6006
