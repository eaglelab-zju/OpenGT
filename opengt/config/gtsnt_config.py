from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('gtsnt')
def set_cfg_gtsnt(cfg):
    """GT-SNT-only hyperparameters (see https://github.com/Zhhuizhe/GT-SNT).

    Shared stack depth / heads / dropout use ``cfg.gt.*`` per OpenGT convention.
    """
    cfg.gtsnt = CN()

    cfg.gtsnt.cb_channels = 8
    cfg.gtsnt.T = 3
    cfg.gtsnt.neuron = 'PLIF'
    cfg.gtsnt.v_threshold = 1.0
    cfg.gtsnt.init_beta = 0.2

    # >0 caps codebook size; <=0 or unset in YAML means no cap (None at runtime).
    cfg.gtsnt.maximum_codes_num = -1

    cfg.gtsnt.normalize = False
    cfg.gtsnt.enable_residual = False
    cfg.gtsnt.enable_norm = False
