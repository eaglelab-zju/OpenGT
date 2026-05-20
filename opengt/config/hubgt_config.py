from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('hubgt')
def set_cfg_hubgt(cfg):
    """HubGT-only knobs. Shared transformer hyperparameters use ``cfg.gt`` (``layers``,
    ``n_heads``, ``dropout``, ``attn_dropout``, ``dim_hidden``) like other GT models.
    """
    cfg.hubgt = CN()

    cfg.hubgt.dp_input = 0.1
    cfg.hubgt.dp_bias = 0.0
    cfg.hubgt.ffn_ratio = 4.0
    cfg.hubgt.num_global_node = 0
    # Dense hop-bias is O(N^2); default must cover PubMed/Actor (~20k nodes) used in benchmarks.
    cfg.hubgt.max_nodes = 65536
