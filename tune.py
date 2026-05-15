import datetime
import os
import torch
import logging
import argparse
import json
import sys
import time

import opengt  # noqa, register custom modules
from opengt.agg_runs import agg_runs
from opengt.optimizer.extra_optimizers import ExtendedSchedulerConfig

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything

from opengt.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from opengt.logger import create_logger


torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True


def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum)


def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode, eval_period=cfg.train.eval_period)


def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def custom_set_run_dir(cfg, run_id):
    """Custom output directory naming for each experiment run.

    Args:
        cfg (CfgNode): Configuration node
        run_id (int): Main for-loop iter id (the random seed or dataset split)
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)


def run_loop_settings():
    """Create main loop execution settings based on the current cfg.

    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from
        the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random
        seed is reset to the initial cfg.seed value for each run iteration.

    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    if len(cfg.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        #seeds = [cfg.seed + x for x in range(num_iterations)]
        seeds = [x for x in range(num_iterations)] # use seeds starting from 0 when tuning
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices

import optuna


def parse_tune_cli():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--optuna_n_trials', type=int, default=20)
    parser.add_argument('--optuna_storage', type=str,
                        default='sqlite:///my_study.db')
    parser.add_argument('--optuna_study_suffix', type=str, default='')
    parser.add_argument('--optuna_sampler_seed', type=int, default=0)
    parser.add_argument('--tune_mode', type=str, default='auto',
                        choices=['auto', 'legacy', 'puregnn'])
    return parser.parse_known_args()


def objective(trial):
    global cfg, args, out_dir_base, tune_args
    
    cfg.out_dir = os.path.join(out_dir_base, "trial_"+str(trial.number))

    # Modify config based on the trial
    cfg.optim.base_lr = trial.suggest_float('base_lr', 1e-5, 1e-2, log=True)
    cfg.optim.weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    tune_mode = tune_args.tune_mode
    if tune_mode == 'auto':
        tune_mode = 'puregnn' if cfg.model.type == 'PureGNN' else 'legacy'

    if tune_mode == 'puregnn':
        cfg.gt.layers = trial.suggest_int('num_layers', 2, 8)
        cfg.gt.dim_hidden = trial.suggest_categorical(
            'dim_hidden', [64, 96, 128, 192, 256, 384, 512]
        )
        cfg.gnn.dim_inner = cfg.gt.dim_hidden
        cfg.gnn.dropout = trial.suggest_float('dropout', 0.0, 0.8, step=0.1)
        cfg.gnn.act = trial.suggest_categorical('act', ['relu', 'gelu'])
        cfg.gnn.batchnorm = trial.suggest_categorical('batchnorm', [True, False])
        cfg.optim.optimizer = trial.suggest_categorical('optimizer', ['adam', 'adamW'])
        cfg.optim.scheduler = trial.suggest_categorical(
            'scheduler', ['none', 'cosine_with_warmup']
        )
        if cfg.optim.scheduler == 'cosine_with_warmup':
            cfg.optim.num_warmup_epochs = trial.suggest_int('num_warmup_epochs', 5, 40, step=5)
        else:
            cfg.optim.num_warmup_epochs = 0
    elif 'Graphormer' in cfg.model.type:
        cfg.graphormer.num_layers = trial.suggest_int('num_layers', 1, 8)
        cfg.graphormer.embed_dim  = trial.suggest_int('dim_hidden', 24, 144, step=24)
        cfg.gnn.dim_inner = cfg.graphormer.embed_dim
        cfg.graphormer.dropout = trial.suggest_float('dropout', 0, 0.8, step=0.1)
        cfg.graphormer.num_heads = trial.suggest_int('n_heads', 1, 4)
    else:
        cfg.gt.layers = trial.suggest_int('num_layers', 1, 8)
        cfg.gt.dim_hidden = trial.suggest_int('dim_hidden', 24, 144, step=24)
        cfg.gnn.dim_inner = cfg.gt.dim_hidden
        cfg.gt.dropout = trial.suggest_float('dropout', 0, 0.8, step=0.1)
        cfg.gt.n_heads = trial.suggest_int('n_heads', 1, 4)
        cfg.gt.attn_dropout = trial.suggest_float('attn_dropout', 0, 0.8, step=0.1)
    if 'DIF' in cfg.model.type:
        cfg.gt.graph_weight = trial.suggest_float('graph_weight', 0.1, 0.9, step=0.1)
    if 'SG' in cfg.model.type:
        cfg.gt.graph_weight = trial.suggest_float('graph_weight', 0.1, 0.9, step=0.1)
        cfg.gt.aggregate = trial.suggest_categorical('aggregate', ['add', 'cat'])
    if 'DeGTA' in cfg.model.type:
        cfg.gt.K = trial.suggest_categorical('K', [2, 4, 8])


    dump_cfg(cfg)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    # Repeat for multiple experiment runs
    for run_id, seed, split_index in zip(*run_loop_settings()):
        # Set configurations for each run
        custom_set_run_dir(cfg, run_id)
        set_printing()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        auto_select_device()
        if cfg.pretrained.dir:
            cfg = load_pretrained_model_cfg(cfg)
        logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}, "
                     f"split_index={cfg.dataset.split_index}")
        logging.info(f"    Starting now: {datetime.datetime.now()}")
        # Set machine learning pipeline
        logging.info(f"   Create Loader: {datetime.datetime.now()}")
        preprocess_start = time.perf_counter()
        loaders = create_loader()
        preprocess_time_s = time.perf_counter() - preprocess_start
        logging.info(f"   Preprocess+Loader time: {preprocess_time_s:.2f}s")
        logging.info(f"   Create Logger: {datetime.datetime.now()}")
        loggers = create_logger()
        for logger in loggers:
            if hasattr(logger, 'set_runtime_stats'):
                logger.set_runtime_stats(
                    preprocess_time_s=round(preprocess_time_s, cfg.round))
        logging.info(f"   Create Model: {datetime.datetime.now()}")
        model = create_model()
        if cfg.pretrained.dir:
            model = init_model_from_pretrained(
                model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
                cfg.pretrained.reset_prediction_head, seed=cfg.seed
            )
        optimizer = create_optimizer(model.parameters(),
                                     new_optimizer_config(cfg))
        scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)
        # Start training
        if cfg.train.mode == 'standard':
            if cfg.wandb.use:
                logging.warning("[W] WandB logging is not supported with the "
                                "default train.mode, set it to `custom`")
            datamodule = GraphGymDataModule()
            train(model, datamodule, logger=True)
        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                       scheduler)
    # Aggregate results from different seeds
    res = 0
    try:
        res = agg_runs(cfg.out_dir, cfg.metric_best)
        print(f"Final result for this trial: {res}")
    except Exception as e:
        logging.info(f"Failed when trying to aggregate multiple runs: {e}")
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
    logging.info(f"[*] All done: {datetime.datetime.now()}")

    return res


if __name__ == '__main__':
    tune_args, remaining_argv = parse_tune_cli()
    sys.argv = [sys.argv[0]] + remaining_argv
    
    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    out_dir_base = cfg.out_dir
    node_encoder_name = cfg.dataset.node_encoder_name
    if not cfg.dataset.node_encoder:
        node_encoder_name = 'none'
    layer_name = getattr(cfg.gt, 'layer_type', 'none')
    suffix = f"_{tune_args.optuna_study_suffix}" if tune_args.optuna_study_suffix else ""
    study_name = (
        f"my_study_{cfg.model.type}+{node_encoder_name}_{cfg.dataset.name}_{layer_name}{suffix}"
    )
    study = optuna.create_study(study_name=study_name,
                                storage=tune_args.optuna_storage,
                                direction=('minimize' if cfg.metric_agg == 'argmin' else 'maximize'),
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=tune_args.optuna_sampler_seed))
    study.optimize(objective, n_trials=tune_args.optuna_n_trials)

    best_summary = {
        "study_name": study_name,
        "best_value": study.best_value,
        "best_trial_number": study.best_trial.number,
        "best_params": study.best_params,
        "dataset": cfg.dataset.name,
        "model_type": cfg.model.type,
        "layer_type": layer_name,
    }
    os.makedirs(out_dir_base, exist_ok=True)
    with open(os.path.join(out_dir_base, "optuna_best.json"), "w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=2, ensure_ascii=False)