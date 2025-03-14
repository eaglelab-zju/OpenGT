#!/usr/bin/env bash

MODEL=SpecFormer
CONFIG=wn-chameleon-SpecFormer

python main.py --cfg configs/${MODEL}/${CONFIG}.yaml

tensorboard --logdir=results/${CONFIG} --port=6006
