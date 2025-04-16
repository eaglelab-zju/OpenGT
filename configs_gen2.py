import argparse
import copy
import csv
import os
import os.path as osp
import random

import numpy as np
import yaml

import torch_geometric.graphgym.contrib  # noqa
from torch_geometric.graphgym.utils.comp_budget import match_baseline_cfg
from torch_geometric.graphgym.utils.io import makedirs_rm_exist, string_to_python

random.seed(123)


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        dest='model',
                        help='model name',
                        default=None,
                        type=str)
    parser.add_argument('--dataset0',
                        dest='dataset0',
                        help='original dataset name',
                        required=True,
                        type=str)
    parser.add_argument('--dataset',
                        dest='dataset',
                        help='dataset name',
                        required=True,
                        type=str)
    return parser.parse_args()


def get_fname(string):
    if string is not None:
        return string.split('/')[-1].split('.')[0]
    else:
        return 'default'

def load_config(fname):
    if fname is not None:
        with open(fname) as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    else:
        return {}

args = parse_args()
config_name=f'configs/{args.model}/{args.dataset0}-{args.model}.yaml'
config = load_config(config_name)

config_out = copy.deepcopy(config)
config_out['wandb']['use'] = False
config_out['wandb']['project'] = args.dataset
config_out['dataset']['name'] = args.dataset

config_out['dataset']['split_mode'] = 'standard'

if args.dataset in ['cora','citeseer','pubmed']:
    config_out['dataset']['format'] = 'PyG-Planetoid'
    config_out['dataset']['task'] = 'node'
    config_out['dataset']['task_type'] = 'classification'
    config_out['model']['loss_fun'] = 'cross_entropy'
    config_out['gnn']['head'] = 'node'
if args.dataset in ['actor']:
    config_out['dataset']['format'] = 'PyG-Actor'
    config_out['dataset']['name'] = 'none'
    config_out['dataset']['task'] = 'node'
    config_out['dataset']['task_type'] = 'classification'
    config_out['model']['loss_fun'] = 'cross_entropy'
    config_out['gnn']['head'] = 'node'
if args.dataset in ['chameleon', 'squirrel']:
    config_out['dataset']['format'] = 'PyG-WikipediaNetwork'
    config_out['dataset']['task'] = 'node'
    config_out['dataset']['task_type'] = 'classification'
    config_out['model']['loss_fun'] = 'cross_entropy'
    config_out['gnn']['head'] = 'node'




with open(f'configs/{args.model}/{args.dataset}-{args.model}.yaml', 'w') as f:
    yaml.dump(config_out, f, default_flow_style=False)
print(f'new configuration saved to: configs/{args.model}/{args.dataset}-{args.model}.yaml')