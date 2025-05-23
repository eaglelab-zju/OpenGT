# OpenGT

Official code for [OpenGT](https://openreview.net/forum?id=qa1nx4CYID), a comprehensive and extensible benchmark suite built on top of [`torch_geometric.graphgym`](https://github.com/pyg-team/pytorch_geometric/tree/master/graphgym), designed for evaluating and analyzing **Graph Transformers (GTs)** under a unified framework.

## Overview

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


## Running Experiment
```bash
conda activate opengt

# Use main.py with config file to run
# Config files are placed at "configs/<ModelName>/<DatasetName-ModelName>.yaml"
# For example, to run DIFFormer model on cora dataset 3 times, use:

python main.py --cfg configs/DIFFormer/cora-DIFFormer.yaml --repeat 3

# then the results will be stored into results/DIFFormer/cora-DIFFormer/

# Use run_batch.sh to tune hyperparameters with grid file at "grids/"

# Use agg_test.py to obtain aggregated results for all experiments

```

