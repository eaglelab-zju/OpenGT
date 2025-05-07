#!/bin/bash

# Define the list of models and datasets
#models=("DIFFormer" "NodeFormer" "GPS" "GPS+RWSE" "GPS+GE" "Graphormer" "GRIT" "SGFormer" "SAN" "SpecFormer" "Exphormer" "Graphtransformer")
#models=("DIFFormer" "NodeFormer" "Graphormer" "GRIT" "SGFormer" "SpecFormer" "Graphtransformer")
models=("SGFormer")

datasets=("cornell" "texas" "wisconsin" "chameleon" "squirrel" "cora" "citeseer" "actor")
#datasets=("ogbg-molhiv" "ogbg-molpcba")


# Loop through each model and dataset
for model in "${models[@]}"; do
	for dataset in "${datasets[@]}"; do
		# Generate the configuration file
		python configs_gen2.py --model "$model" --dataset "$dataset" --dataset0 wn-chameleon --accelerator cuda:1
		
		# Run the main script with the generated configuration
		python main.py --repeat 3 --cfg "configs/$model/$dataset-$model.yaml"
	done
done
