#!/bin/bash

# Define the list of models and datasets
# models=("DIFFormer" "NodeFormer" "GPS+RWSE" "Graphormer" "GRIT" "SGFormer" "SAN" "SpecFormer")
models=("Exphormer")
datasets=("cora" "citeseer" "pubmed" "actor")

# Loop through each model and dataset
for model in "${models[@]}"; do
	for dataset in "${datasets[@]}"; do
		# Generate the configuration file
		python configs_gen2.py --model "$model" --dataset "$dataset" --dataset0 wn-chameleon
		
		# Run the main script with the generated configuration
		python main.py --cfg "configs/$model/$dataset-$model.yaml" --repeat 3
	done
done
