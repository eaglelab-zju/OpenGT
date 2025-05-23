from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


def set_cfg_preprocess(cfg):
    """Extend configuration with preprocessing options
    """

    cfg.prep = CN()

    # Argument group for adding expander edges

    # if it's enabled expander edges would be available by e.g. data.expander_edges
    cfg.prep.exp = False
    cfg.prep.exp_algorithm = 'Random-d' # Other option is 'Hamiltonian'
    cfg.prep.use_exp_edges = True
    cfg.prep.exp_deg = 5
    cfg.prep.exp_max_num_iters = 100
    cfg.prep.add_edge_index = True
    cfg.prep.num_virt_node = 0
    cfg.prep.exp_count = 1
    #cfg.prep.add_self_loops = False
    #cfg.prep.add_reverse_edges = True
    #cfg.prep.train_percent = 0.6
    #cfg.prep.layer_edge_indices_dir = None



    # Argument group for adding node distances
    cfg.prep.dist_enable = False
    cfg.prep.dist_cutoff = 510

    # Multihop preprocess for Nodeformer, need to be same with gt.rb_order
    cfg.prep.rb_order = 1 

    # METIS partitioning
    cfg.metis = CN()
    cfg.metis.enable = False
    cfg.metis.patches = 0
    cfg.metis.num_hops = 1
    cfg.metis.drop_rate = 0.3
    cfg.metis.online = True
    cfg.metis.patch_rw_dim = 0
    cfg.metis.patch_num_diff = -1


register_config('preprocess', set_cfg_preprocess)
