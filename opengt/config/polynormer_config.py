from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('polynormer')
def set_cfg_polynormer(cfg):
    """Polynormer-only knobs. Local stack depth follows ``cfg.gt.layers`` like other
    transformer-style stacks; global branch depth stays under ``cfg.polynormer.global_layers``.
    """
    cfg.polynormer = CN()

    # Global transformer depth when ``use_global`` is True (PyG default-style).
    cfg.polynormer.global_layers = 2
    cfg.polynormer.use_global = True

    cfg.polynormer.in_dropout = 0.15
    cfg.polynormer.dropout = 0.2
    cfg.polynormer.global_dropout = 0.2
    cfg.polynormer.heads = 1
    cfg.polynormer.beta = 0.9
    cfg.polynormer.qk_shared = False
    cfg.polynormer.pre_ln = False
    cfg.polynormer.post_bn = True
    cfg.polynormer.local_attn = False

