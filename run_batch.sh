#!/usr/bin/env bash

CONFIG=wn-chameleon-SpecFormer
GRID=wn-chameleon-SpecFormer
REPEAT=3
MAX_JOBS=8

# generate configs (after controlling computational budget)
# please remove --config_budget, if don't control computational budget
#  --config_budget configs/${CONFIG}.yaml \

python configs_gen.py --config configs/SpecFormer/${CONFIG}.yaml \
  --grid grids/${GRID}.txt \
  --out_dir configs/SpecFormer
#python configs_gen.py --config configs/ChemKG/${CONFIG}.yaml --config_budget configs/ChemKG/${CONFIG}.yaml --grid grids/ChemKG/${GRID}.txt --out_dir configs
# run batch of configs
# Args: config_dir, num of repeats, max jobs running
bash parallel.sh configs/SpecFormer/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS
# rerun missed / stopped experiments
bash parallel.sh configs/SpecFormer/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS
# rerun missed / stopped experiments
bash parallel.sh configs/SpecFormer/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS

# aggregate results for the batch
python agg_batch.py --dir results/${CONFIG}_grid_${GRID}