# PyTorch 2.6+ defaults torch.load(weights_only=True). OGB / PyG processed pickles and
# many checkpoints omit the kwarg; unpickling then fails unless we default to False here
# (local dataset caches are trusted in this repo).
import torch as _torch

if not getattr(_torch, "_opengt_torch_load_patched", False):
    _opengt_real_torch_load = _torch.load

    def _opengt_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            try:
                return _opengt_real_torch_load(*args, **kwargs, weights_only=False)
            except TypeError:
                return _opengt_real_torch_load(*args, **kwargs)
        return _opengt_real_torch_load(*args, **kwargs)

    _torch.load = _opengt_torch_load
    _torch._opengt_torch_load_patched = True

# Optional: allowlist PyG types when callers explicitly use weights_only=True.
try:
    import torch.serialization as _ts
    from torch_geometric.data.data import DataEdgeAttr as _DataEdgeAttr

    _ts.add_safe_globals([_DataEdgeAttr])
except Exception:  # pragma: no cover - optional depending on torch/pyg versions
    pass

from .act import *  # noqa
from .config import *  # noqa
from .encoder import *  # noqa
from .head import *  # noqa
from .layer import *  # noqa
from .loader import *  # noqa
from .loss import *  # noqa
from .network import *  # noqa
from .optimizer import *  # noqa
from .pooling import *  # noqa
from .stage import *  # noqa
from .train import *  # noqa
from .transform import *  # noqa
