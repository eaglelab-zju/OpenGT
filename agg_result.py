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
			if dataset in ["ogbg-molhiv"]:
				crit = "auc"
				better = lambda x, y: x > y  # Higher is better for AUC
			elif dataset in ["ogbg-molpcba", "peptides-func"]:
				crit = "ap"
				better = lambda x, y: x > y  # Higher is better for AP
			elif dataset in ["zinc", "peptides-struct"]:
				crit = "mae"
				better = lambda x, y: x < y  # Lower is better for MSE/MAE
			else:
				crit = "accuracy"
				better = lambda x, y: x > y  # Higher is better for accuracy
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

		process = lambda x: x if x == "N/A" else str(round(float(x),4))

		# Record metrics
		if model not in data:
			data[model] = {}
		data[model][dataset] = {
			"acc": process(metrics.get("accuracy", "N/A")),
			"acc_std": process(metrics.get("accuracy_std", "N/A")),
			"f1": process(metrics.get("f1", "N/A")),
			"f1_std": process(metrics.get("f1_std", "N/A")),
			"mse": process(metrics.get("mse", "N/A")),
			"mse_std": process(metrics.get("mse_std", "N/A")),
			"mae": process(metrics.get("mae", "N/A")),
			"mae_std": process(metrics.get("mae_std", "N/A")),
			"auc": process(metrics.get("auc", "N/A")),
			"auc_std": process(metrics.get("auc_std", "N/A")),
			"ap": process(metrics.get("ap", "N/A")),
			"ap_std": process(metrics.get("ap_std", "N/A")),
		}

		# Collect time statistics
		times = []
		for run_id in os.listdir(folder_path):
			if not run_id.isdigit():
				continue
			run_path = os.path.join(folder_path, run_id)
			val_stats_path = os.path.join(run_path, "val", "stats.json")
			train_stats_path = os.path.join(run_path, "train", "stats.json")

			if not os.path.exists(val_stats_path) or not os.path.exists(train_stats_path):
				print(f"Skipping run '{run_id}' of model '{model}' on dataset '{dataset}' due to missing stats.json files.")
				continue

			# Get the best epoch from val/stats.json
			with open(val_stats_path, "r") as f:
				best_epoch = 0
				best_value = None
				for line in f:
					try:
						val_data = json.loads(line)
						if val_data.get(crit) is not None:
							if best_value is None or better(val_data[crit], best_value):
								best_value = val_data[crit]
								best_epoch = val_data.get("epoch", 0)
						else:
							print(f"Missing '{crit}' in val stats for run '{run_id}' of model '{model}' on dataset '{dataset}'.")
							continue
					except json.JSONDecodeError:
						print(f"Skipping invalid JSON line in val stats for run '{run_id}' of model '{model}' on dataset '{dataset}'.")
						continue
				if best_epoch is None:
					print(f"Missing 'best_epoch' in val stats for run '{run_id}' of model '{model}' on dataset '{dataset}'.")
					continue

			# Calculate total time up to the best epoch from train/stats.json
			with open(train_stats_path, "r") as f:
				epoch_times = []
				for line in f:
					try:
						epoch_data = json.loads(line)
						epoch_times.append(epoch_data.get("time_epoch", 0))
					except json.JSONDecodeError:
						print(f"Skipping invalid JSON line in train stats for run '{run_id}' of model '{model}' on dataset '{dataset}'.")
						continue

				if len(epoch_times) <= best_epoch:
					print(f"Insufficient epoch times in train stats for run '{run_id}' of model '{model}' on dataset '{dataset}'.")
					continue

			total_time = sum(epoch_times[:best_epoch + 1])
			times.append(total_time)

		# Calculate average time and standard deviation
		if times:
			avg_time = sum(times) / len(times)
			std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
			avg_time = round(avg_time, 4)
			std_time = round(std_time, 4)
			data[model][dataset]["time"] = avg_time
			data[model][dataset]["time_std"] = std_time
		else:
			data[model][dataset]["time"] = "N/A"
			data[model][dataset]["time_std"] = "N/A"

# Write data to separate CSV files for each metric
metrics_to_write = ["acc", "f1", "mse", "mae", "auc", "ap", "time"]
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
