#!/usr/bin/env bash

MODEL=DeGTA
CONFIG=wn-chameleon-DeGTA
GRID=wn-chameleon-DeGTA
REPEAT=3
MAX_JOBS=8

# generate configs (after controlling computational budget)
# please remove --config_budget, if don't control computational budget
#  --config_budget configs/${CONFIG}.yaml \

python configs_gen.py --config configs/${MODEL}/${CONFIG}.yaml \
  --grid grids/${GRID}.txt \
  --out_dir configs/${MODEL}
#python configs_gen.py --config configs/ChemKG/${CONFIG}.yaml --config_budget configs/ChemKG/${CONFIG}.yaml --grid grids/ChemKG/${GRID}.txt --out_dir configs
# run batch of configs
# Args: config_dir, num of repeats, max jobs running
bash parallel.sh configs/${MODEL}/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS
# rerun missed / stopped experiments
bash parallel.sh configs/${MODEL}/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS
# rerun missed / stopped experiments
bash parallel.sh configs/${MODEL}/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS

# aggregate results for the batch
python agg_batch.py --dir results/${CONFIG}_grid_${GRID}
