from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('m3dphormer')
def set_cfg_m3dphormer(cfg):
    """M3Dphormer-only options. Layer count / heads / dropout use ``cfg.gt`` (see skill)."""
    cfg.m3dphormer = CN()
    # Defaults aligned with upstream readme example (NeurIPS 2025 code release).
    cfg.m3dphormer.num_clusters = 128
    cfg.m3dphormer.global_nodes_per_class = 1
    cfg.m3dphormer.learn_global = False
    cfg.m3dphormer.use_cache = False
    cfg.m3dphormer.use_res = True
    cfg.m3dphormer.norm_type = 'rms'
    cfg.m3dphormer.norm_pos = 'pre'
    cfg.m3dphormer.local_type = 'GAT'
    cfg.m3dphormer.max_nodes = 8192
    cfg.m3dphormer.use_global_aux_loss = True
    cfg.m3dphormer.global_aux_loss_weight = 1.0
