#!/bin/bash

# Define the list of models and datasets
#models=("DIFFormer" "NodeFormer" "GPS" "GPS+RWSE" "Graphormer" "GRIT" "SGFormer" "SAN" "SpecFormer" "Exphormer")
models=("GPS+GE")

datasets=("cornell" "texas" "wisconsin" "chameleon" "squirrel" "cora" "citeseer" "pubmed" "actor")
#datasets=("squirrel" "pubmed" "citeseer" "actor")

# Loop through each model and dataset
for model in "${models[@]}"; do
	for dataset in "${datasets[@]}"; do
		# Generate the configuration file
		python configs_gen2.py --model "$model" --dataset "$dataset" --dataset0 wn-chameleon --accelerator cuda:0
		
		# Run the main script with the generated configuration
		python main.py --repeat 3 --cfg "configs/$model/$dataset-$model.yaml"
	done
done
