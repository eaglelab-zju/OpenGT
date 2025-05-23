# OpenGT

Official code for [OpenGT](https://openreview.net/forum?id=qa1nx4CYID), a comprehensive and extensible benchmark suite built on top of [`torch_geometric.graphgym`](https://github.com/pyg-team/pytorch_geometric/tree/master/graphgym), designed for evaluating and analyzing **Graph Transformers (GTs)** under a unified framework.

## Overview

The picture below shows a timeline of GT development, dividing GTs into 3 categories.

![GT Timeline](./timeline.png)

## 🔧 Features

- ✅ **Standardized Implementations** of 16 Graph Transformer and GNN baselines, including:
  - GT models: `Graphormer`, `NodeFormer`, `DIFFormer`, `GraphGPS`, `GRIT`, `SpecFormer`, `Exphormer`, `SAN`, `SGFormer`, `CoBFormer`, `GraphMLPMixer`, `GraphTransformer`, `DeGTA`
  - GNN baselines: `GCN`, `GAT`, `APPNP`
- 📊 **Unified Training and Evaluation Pipeline**
  - Consistent data splits and batch sizes per dataset
  - Standardized metric computation and logging
- 🧪 **Flexible Hyperparameter Tuning**
  - Provides an easy-to-use interface for performing grid search over multiple hyperparameters
  - Supports specifying search spaces via configuration files
  - Automatically logs and evaluates results across all combinations for robust comparison
- 📁 **Diverse Datasets**
  - Covers both **node-level** and **graph-level** tasks
  - Includes graphs with varying levels of **homophily** and **sparsity**

## 🎯 Goal

**OpenGT** aims to promote:
- 📌 Fair and reproducible comparisons across Graph Transformers
- 🔍 Deeper understanding of design choices and their practical implications
- 🚀 Acceleration of GT research through a solid benchmarking foundation

## ⚙ Python environment setup with Conda

```bash
conda create -n opengt python=3.10
conda activate opengt

pip install pytorch=2.5 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install torch_geometric

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
conda install openbabel fsspec rdkit

pip install pytorch-lightning yacs torchmetrics
pip install performer-pytorch
pip install tensorboardX
pip install ogb
pip install wandb
pip install pymetis

conda clean --all
```

## 🚀 Running Experiments

### 1. Activate Environment

First, activate the project environment:

```bash
conda activate opengt
```

---

### 2. Run a Single Experiment

Use `main.py` along with a configuration file to launch an experiment. Configuration files are stored in:

```
configs/<ModelName>/<DatasetName-ModelName>.yaml
```

#### Example: Run DIFFormer on Cora (3 runs)

```bash
python main.py --cfg configs/DIFFormer/cora-DIFFormer.yaml --repeat 3
```

Results will be saved automatically to:

```
results/DIFFormer/cora-DIFFormer/
```

---

### 3. Hyperparameter Tuning

To perform grid search on hyperparameters, use the provided `run_batch.sh` script and corresponding grid files under the `grids/` directory.


```bash
bash run_batch.sh
```

Please note that the grid file and the configuration file names should be modified in the script.

Each line of the grid file should be written in the following format:

```text
<Config parameter name> <Display name> <List of possible values>
```

For example:

```text
gt.layers nlayer [1,2,3,4]
gt.aggregate agg ['add', 'cat']
gt.dropout dropout [0.2, 0.5, 0.8]
```

This will explore all parameter combinations defined in the relevant grid configuration.

---

### 4. Aggregating Results

To summarize and aggregate results from multiple runs:

```bash
python agg_test.py
```

This script collects results across seeds and outputs averaged performance metrics with standard deviations in to a `.csv` file.
