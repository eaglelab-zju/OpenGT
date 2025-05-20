#!/bin/bash

# Define the list of models and datasets
#models=("DIFFormer" "NodeFormer" "GPS" "GPS+RWSE" "GPS+GE" "Graphormer" "GRIT" "SGFormer" "SAN" "SpecFormer" "Exphormer" "Graphtransformer" "GraphMLPMixer" "GCN" "GAT" "APPNP")
#models=("DIFFormer" "GRIT" "SGFormer" "SpecFormer" "Graphtransformer" "GraphMLPMixer")
#models=("APPNP" "GCN" "GAT")
models=("Graphtransformer")
#models=("SGFormer+LapPE" "SGFormer+ESLapPE" "SGFormer+RWSE" "SGFormer+GE" "SGFormer+GESP" "SGFormer+WLSE" "DIFFormer+LapPE" "DIFFormer+ESLapPE" "DIFFormer+RWSE" "DIFFormer+GE" "DIFFormer+GESP" "DIFFormer+WLSE" "GPS+None" "GPS+RWSE" "GPS+GE" "GPS+GESP" "GPS+WLSE" "GPS+ESLapPE")

datasets=("cornell" "texas" "wisconsin" "chameleon" "squirrel" "cora" "citeseer" "pubmed" "actor")
#datasets=("cora" "citeseer" "pubmed")
datasets=("peptides-func" "peptides-struct")

# Loop through each model and dataset
for model in "${models[@]}"; do
	for dataset in "${datasets[@]}"; do
		# Generate the configuration file
		python configs_gen2.py --model "$model" --dataset "$dataset" --dataset0 wn-chameleon --accelerator cuda:4
		
		# Run the main script with the generated configuration
		python main.py --repeat 3 --cfg "configs/$model/$dataset-$model.yaml"
	done
done
