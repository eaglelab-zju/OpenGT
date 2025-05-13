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
    parser.add_argument('--accelerator',
                        dest='accelerator',
                        help='accelerator name',
                        default='cuda:0',
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
config_out['accelerator'] = args.accelerator
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
if args.dataset in ['cornell', 'texas', 'wisconsin']:
    config_out['dataset']['format'] = 'PyG-WebKB'
    config_out['dataset']['task'] = 'node'
    config_out['dataset']['task_type'] = 'classification'
    config_out['model']['loss_fun'] = 'cross_entropy'
    config_out['gnn']['head'] = 'node'
if args.dataset in ['zinc']:
    config_out['metric_best'] = 'mae'
    config_out['metric_agg'] = 'argmin'
    config_out['dataset']['format'] = 'PyG-ZINC'
    config_out['dataset']['name'] = 'subset'
    config_out['dataset']['task'] = 'graph'
    config_out['dataset']['task_type'] = 'regression'
    config_out['dataset']['transductive'] = False
    config_out['dataset']['node_encoder'] = True
    config_out['dataset']['node_encoder_name'] = 'TypeDictNode'
    config_out['dataset']['node_encoder_num_types'] = 28
    config_out['dataset']['edge_encoder'] = True
    config_out['dataset']['edge_encoder_name'] = 'TypeDictEdge'
    config_out['dataset']['edge_encoder_num_types'] = 4
    config_out['train']['batch_size'] = 32
    config_out['train']['eval_period'] = 1
    config_out['train']['ckpt_period'] = 100
    config_out['model']['loss_fun'] = 'l1'
    config_out['model']['edge_decoding'] = 'dot'
    config_out['model']['graph_pooling'] = 'add'
    config_out['gt']['layers'] = 8
    config_out['gt']['n_heads'] = 4
    config_out['gt']['batch_norm'] = True
    config_out['gnn']['head'] = 'san_graph'
    config_out['optim']['max_epoch'] = 2000
    config_out['optim']['scheduler'] = 'cosine_with_warmup'
    config_out['optim']['num_warmup_epochs'] = 50
if args.dataset in ['ogbg-molhiv', 'ogbg-molpcba']:
    if args.dataset == 'ogbg-molhiv':
        config_out['metric_best'] = 'auc'
        config_out['dataset']['task_type'] = 'classification'
        config_out['train']['batch_size'] = 32
    else:
        config_out['metric_best'] = 'ap'
        config_out['dataset']['task_type'] = 'classification_multilabel'
        config_out['train']['batch_size'] = 512
    config_out['metric_agg'] = 'argmax'
    config_out['dataset']['format'] = 'OGB'
    config_out['dataset']['name'] = args.dataset
    config_out['dataset']['task'] = 'graph'
    config_out['dataset']['transductive'] = False
    config_out['dataset']['node_encoder'] = True
    config_out['dataset']['node_encoder_name'] = 'Atom'
    config_out['dataset']['edge_encoder'] = True
    config_out['dataset']['edge_encoder_name'] = 'Bond'
    config_out['train']['eval_period'] = 1
    config_out['train']['ckpt_period'] = 100
    config_out['model']['loss_fun'] = 'cross_entropy'
    config_out['model']['edge_decoding'] = 'dot'
    config_out['model']['graph_pooling'] = 'mean'
    if 'gt' in config_out:
        config_out['gt']['layers'] = 8
        config_out['gt']['n_heads'] = 4
        config_out['gt']['batch_norm'] = True
    config_out['gnn']['head'] = 'san_graph'
    config_out['optim']['max_epoch'] = 100
    config_out['optim']['num_warmup_epochs'] = 10
if args.dataset in ['peptides-func', 'peptides-struct']:
    if args.dataset == 'peptides-func':
        config_out['metric_best'] = 'ap'
        config_out['metric_agg'] = 'argmax'
        config_out['dataset']['task_type'] = 'classification_multilabel'
        config_out['dataset']['name'] = 'peptides-functional'
    else:
        config_out['metric_best'] = 'mae'
        config_out['metric_agg'] = 'argmin'
        config_out['dataset']['task_type'] = 'regression'
        config_out['dataset']['name'] = 'peptides-structural'
    config_out['dataset']['format'] = 'OGB'
    config_out['dataset']['task'] = 'graph'
    config_out['dataset']['transductive'] = False
    config_out['dataset']['node_encoder'] = True
    config_out['dataset']['node_encoder_name'] = 'Atom'
    config_out['dataset']['node_encoder_num_types'] = 20
    config_out['dataset']['edge_encoder'] = True
    config_out['dataset']['edge_encoder_name'] = 'Bond'
    config_out['dataset']['edge_encoder_num_types'] = 4
    config_out['train']['eval_period'] = 1
    config_out['train']['ckpt_period'] = 100
    config_out['train']['batch_size'] = 128
    if args.dataset == 'peptides-func':
        config_out['model']['loss_fun'] = 'cross_entropy'
        config_out['model']['graph_pooling'] = 'mean'
    else:
        config_out['model']['loss_fun'] = 'l1'
        config_out['model']['graph_pooling'] = 'mean'
    if 'gt' in config_out:
        config_out['gt']['layers'] = 4
        config_out['gt']['n_heads'] = 4
        config_out['gt']['batch_norm'] = True
    config_out['gnn']['head'] = 'default'
    config_out['optim']['max_epoch'] = 200
    config_out['optim']['num_warmup_epochs'] = 10

if args.model == 'GraphMLPMixer':
    config_out['gnn']['head'] = 'mlp_mixer_graph'

if args.dataset in ['zinc', 'ogbg-molhiv', 'ogbg-molpcba', 'peptides-func', 'peptides-struct']:
    if len(args.model.split('+'))>1:
        pe_name = args.model.split('+')[1]
        if pe_name == 'GE':
            pe_name = 'GraphormerBias'
        config_out['dataset']['node_encoder_name'] += '+'+args.model.split('+')[1]
    elif args.model == 'DeGTA':
        config_out['dataset']['node_encoder_name'] += '+LapPE+RWSE'

with open(f'configs/{args.model}/{args.dataset}-{args.model}.yaml', 'w') as f:
    yaml.dump(config_out, f, default_flow_style=False)
print(f'new configuration saved to: configs/{args.model}/{args.dataset}-{args.model}.yaml')