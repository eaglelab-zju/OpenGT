import os
import json
import csv

# Directory containing the results
results_dir = "results/"

# Initialize data storage
data = {}

# Enumerate all folders under the results directory
for folder_name in os.listdir(results_dir):
	folder_path = os.path.join(results_dir, folder_name)
	if os.path.isdir(folder_path) and "grid" not in folder_name:
		# Split folder name into dataset and model
		try:
			dataset, model = folder_name.rsplit("-", 1)
		except ValueError:
			print(f"Skipping folder with invalid name format: {folder_name}")
			continue

		# Check if "agg" folder exists
		agg_path = os.path.join(folder_path, "agg")
		if not os.path.exists(agg_path):
			print(f"Model '{model}' did not successfully run on dataset '{dataset}'.")
			continue

		# Check for best.json in agg/test/
		best_json_path = os.path.join(agg_path, "test", "best.json")
		if not os.path.exists(best_json_path):
			print(f"Missing 'best.json' for model '{model}' on dataset '{dataset}'.")
			continue

		# Read metrics from best.json
		with open(best_json_path, "r") as f:
			metrics = json.load(f)

		# Record metrics
		if model not in data:
			data[model] = {}
		data[model][dataset] = {
			"acc": metrics.get("accuracy", "N/A"),
			"acc_std": metrics.get("accuracy_std", "N/A"),
			"f1": metrics.get("f1", "N/A"),
			"f1_std": metrics.get("f1_std", "N/A"),
			"mse": metrics.get("mse", "N/A"),
			"mse_std": metrics.get("mse_std", "N/A"),
			"mae": metrics.get("mae", "N/A"),
			"mae_std": metrics.get("mae_std", "N/A"),
			"auc": metrics.get("auc", "N/A"),
			"auc_std": metrics.get("auc_std", "N/A"),
		}

# Write data to separate CSV files for each metric
metrics_to_write = ["acc", "f1", "mse", "mae", "auc"]
for metric in metrics_to_write:
	output_csv = f"results_summary_{metric}.csv"
	with open(output_csv, "w", newline="") as csvfile:
		writer = csv.writer(csvfile)
		# Write header
		header = ["Model/Dataset"] + sorted({dataset for model_data in data.values() for dataset in model_data})
		writer.writerow(header)

		# Write rows
		for model, datasets in data.items():
			row = [model]
			for dataset in header[1:]:
				if dataset in datasets:
					row.append(f"{datasets[dataset][metric]}±{datasets[dataset][f'{metric}_std']}")
				else:
					row.append("N/A")
			writer.writerow(row)

	print(f"Results summary for {metric} written to {output_csv}")
