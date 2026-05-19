# OpenGT 已接入模型说明（查阅用）

本文档汇总本仓库内 **Polynormer、HubGT、GT-SNT、M3Dphormer、BareTransformer** 的代码位置、配置约定与依赖，便于复现与二次开发。  
接入流程与 checklist 仍以 Cursor skill **`add-opengt-model`** 为准；本文件不承担流程教学，仅作**模型级**说明。

---

## 通用约定（GraphGym）

- **表征 → 分类头**：骨干只输出节点表征；**logits 一律由** `post_mp`（如 `GNNNodeHead`）产生；骨干末端不得再做 `log_softmax` 或与 `post_mp` 重复的分类线性层。
- **`cfg.gt.*`（多模型共用）**：层数、头数、隐维、dropout 等与「Graph Transformer 类」栈共用的超参，优先放在 **`cfg.gt`**（如 `gt.layers`、`gt.n_heads`、`gt.dim_hidden`、`gt.dropout`、`gt.attn_dropout`），各 `opengt/network/*.py` 中统一读取，避免换模型时口径不一致。
- **`cfg.<model>.*`（模型独有）**：仅当其它网络没有或语义不同时，再增加独立配置组（如 `cfg.polynormer`、`cfg.hubgt`、`cfg.gtsnt`、`cfg.m3dphormer`）。
- **任务 YAML**：以 `dataset`、`model.type`、`gnn` / `gt` 对齐维度、`optim` 等为主；模型独有键仅在偏离 Python 默认时再写。

---

## Polynormer

| 项 | 路径 |
|----|------|
| 全局注意力层 | `opengt/layer/polynormer_attention.py`（`register_layer('polynormer_attention', …)`） |
| GraphGym 网络 | `opengt/network/polynormer.py`（`@register_network("Polynormer")`） |
| 默认超参 | `opengt/config/polynormer_config.py` |
| 示例配置 | `configs/Polynormer/cora-Polynormer.yaml`、`configs/Polynormer/chameleon-new-Polynormer.yaml` |

**架构要点**

- `FeatureEncoder` → 局部 GCN/GAT 栈 → 可选 `to_dense_batch` + `PolynormerAttention` 全局分支（由 `cfg.polynormer.use_global` 控制）。
- 若 `cfg.polynormer.two_stage=true`，模型会在 `train()` 内部复刻上游训练日程：前 `cfg.polynormer.local_epochs` 轮强制 local-only，随后切到 global 分支；示例配置默认 **100 + 200** 轮，`optim.max_epoch=300`。
- **局部层数**读 **`cfg.gt.layers`**；**全局分支层数**读 **`cfg.polynormer.global_layers`**。
- `post_mp` 的 `dim_in` 为 **`heads * hidden`**（`inner_channels`），与 `cfg.gnn.dim_inner` 在数学上可区分（多 head concat 时）。
- 单图 transductive 且 `batch` 缺失时，需全 0 `batch` 向量（见 `polynormer.py`）。

**上游**：PyTorch Geometric `nn.models.Polynormer`（MIT），本仓库已 vendored 并适配 GraphGym。

---

## BareTransformer

| 项 | 路径 |
|----|------|
| GraphGym 网络 | `opengt/network/bare_transformer.py`（`@register_network('BareTransformer')`） |
| 示例配置 | `configs/BareTransformer/cora-BareTransformer+LapPE.yaml`、`configs/BareTransformer/chameleon-new-BareTransformer+LapPE.yaml` |

**架构要点**

- `FeatureEncoder`（+ 可选 `GNNPreMP`）后，直接把节点按 `batch` 打包成 dense 序列，使用 PyTorch `nn.MultiheadAttention` 做 **全局 self-attention**。
- **完全不使用 `edge_index` / `edge_attr` 进行消息传递**；图结构只会在 PE 预处理或 node encoder 中间接出现，适合做 LapPE/RWSE/WLSE 等 PE 消融。
- 层数、头数、hidden、dropout、attention dropout、norm/residual 均复用 **`cfg.gt.*`**。
- 输出仍为节点表征，分类由 `post_mp` 完成。

---

## HubGT

| 项 | 路径 |
|----|------|
| 结构块（注意力 + 结构偏置嵌入） | `opengt/layer/hubgt_modules.py` |
| GraphGym 网络 | `opengt/network/hubgt.py`（`@register_network('HubGT')`） |
| 默认超参 | `opengt/config/hubgt_config.py` |
| 示例配置 | `configs/HubGT/cora-HubGT.yaml`、`configs/HubGT/chameleon-new-HubGT.yaml` |

**架构要点**

- 全图稠密自注意力；**注意力偏置**由无向无权最短路跳数矩阵得到（不可达为 `INF8=255`，供 `StructuralEmbedding` mask）。
- **层数 / 头数 / dropout**：**`cfg.gt.layers`、`cfg.gt.n_heads`、`cfg.gt.dropout`、`cfg.gt.attn_dropout`**。
- **`cfg.hubgt`**：`dp_input`、`dp_bias`、`ffn_ratio`、`num_global_node`、`max_nodes` 等；**不**再镜像 `gt.layers`。
- 当前实现要求 **单图 per batch**（Planetoid / Critical full-batch）；多图 mini-batch 未实现。
- 大图注意 **\(O(N^2)\)** 显存与 `hubgt.max_nodes` 上限。

**上游**：[HubGT](https://github.com/gdmnl/HubGT)（MIT），已去掉原图级 `downstream_out_proj`，改为 `post_mp`。

---

## GT-SNT

| 项 | 路径 |
|----|------|
| GCN / SNT / SMHA | `opengt/layer/gtsnt_components.py` |
| GraphGym 网络 | `opengt/network/gtsnt.py`（`@register_network('GTSNT')`） |
| 默认超参 | `opengt/config/gtsnt_config.py` |
| 示例配置 | `configs/GTSNT/cora-GTSNT.yaml`、`configs/GTSNT/chameleon-new-GTSNT.yaml` |

**依赖**

- **`spikingjelly`**（SNT 脉冲神经元）。未安装时 import 会抛出明确提示；可参考 `docs/requirements.txt` 中的注释行安装。
- `torch_sparse` 等 PyG 生态（`gcn_norm` 与 `SNT.propagate` 路径）。

**架构要点**

- 上游 `GTSNT` 的 **`lin_out` 已移除**，分类由 **`post_mp`** 完成。
- 输入在 `FeatureEncoder` 后已是 **`cfg.gnn.dim_inner`**，骨干内 **不再使用 `lin_in`**，层间残差为恒等空间上的 skip。
- **层数 / 头数 / dropout**：**`cfg.gt.layers`、`cfg.gt.n_heads`、`cfg.gt.dropout`、`cfg.gt.attn_dropout`**。
- **`cfg.gtsnt`**：`cb_channels`、`T`、`neuron`、`v_threshold`、`init_beta`、`maximum_codes_num`（≤0 表示不截断 codebook）、`normalize`、`enable_residual`、`enable_norm`。
- **`cfg.share.num_nodes`**：SNT 中可学习 `rand_feat` 形状为 `(N, cb_channels)`。OpenGT 在 **`main.py`** 中于 **`create_loader()` 之后**、**`create_model()` 之前**，根据训练 DataLoader 的首个图写入 `cfg.share.num_nodes`。若在其他入口手动构建模型，须自行设置该字段。

**上游**：[GT-SNT](https://github.com/Zhhuizhe/GT-SNT)（MIT）。

---

## M3Dphormer

| 项 | 路径 |
|----|------|
| 分层掩码 / MoE 注意力核 | `opengt/layer/m3dphormer_modules.py`（vendored 自上游 `networks/M3Dphormer.py`） |
| GraphGym 网络 | `opengt/network/m3dphormer.py`（`@register_network('M3Dphormer')`） |
| 默认超参 | `opengt/config/m3dphormer_config.py` |
| 示例配置 | `configs/M3Dphormer/cora-M3Dphormer.yaml`、`configs/M3Dphormer/chameleon-new-M3Dphormer.yaml` |

**架构要点**

- `FeatureEncoder`（+ 可选 `GNNPreMP`）后得到 **`cfg.gnn.dim_inner`**；将 **METIS 簇节点** 与 **按训练集类原型的全局 token** 拼到特征矩阵，构造与上游一致的 **三级 `edge_masks`**（局部边 / 簇-节点 / 全局-节点）。
- 骨干 **`M3DphormerEncoder`** 不输出 logits；**分类仅由 `post_mp`** 完成（已去掉上游 `cls` 与训练里对 `out` 的 `log_softmax`）。
- **层数 / 头数 / dropout / attn_dropout**：**`cfg.gt.layers`、`cfg.gt.n_heads`、`cfg.gt.dropout`、`cfg.gt.attn_dropout`**。
- **`cfg.m3dphormer`**：`num_clusters`、`global_nodes_per_class`、`learn_global`、`local_type`（`GAT`/`GCN`/`Trans`）、`use_cache`、`use_res`、`norm_type`、`norm_pos`、`max_nodes`、`use_global_aux_loss`、`global_aux_loss_weight`。
- **单图 full-batch**（Planetoid / Critical）；多图 mini-batch 未实现。簇划分使用 **`pymetis`**（与仓库 METIS 预处理一致），**不**依赖上游单独的 `metis` Python 包。
- 上游训练中的 **全局 token 辅助 CE** 已按 `gamma = n_global / num_train` 接入训练 loss；验证/测试 loss 仍只记录真实节点分类 CE，与上游 `evaluate.py` 的记录口径一致。

**上游**：[M3Dphormer](https://github.com/null-xyj/M3Dphormer)（NeurIPS 2025 代码，仓库内无 LICENSE 文件声明）。

---

## 冒烟 / 终验命令示例

```bash
conda run -n opengt python main.py --cfg configs/Polynormer/cora-Polynormer.yaml --repeat 1 seed 42 \
  accelerator cpu optim.max_epoch 3 train.eval_period 1 out_dir results_smoke name_tag p

conda run -n opengt python main.py --cfg configs/HubGT/cora-HubGT.yaml --repeat 1 seed 42 \
  accelerator cpu optim.max_epoch 3 train.eval_period 1 out_dir results_smoke name_tag h

conda run -n opengt python main.py --cfg configs/GTSNT/cora-GTSNT.yaml --repeat 1 seed 42 \
  accelerator cpu optim.max_epoch 3 train.eval_period 1 out_dir results_smoke name_tag g

conda run -n opengt python main.py --cfg configs/M3Dphormer/cora-M3Dphormer.yaml --repeat 1 seed 42 \
  accelerator cpu optim.max_epoch 3 train.eval_period 1 out_dir results_smoke name_tag m3d

conda run -n opengt python main.py --cfg configs/BareTransformer/cora-BareTransformer+LapPE.yaml --repeat 1 seed 42 \
  accelerator cpu optim.max_epoch 3 train.eval_period 1 out_dir results_smoke name_tag bt
```

终验建议：空闲 GPU、`optim.max_epoch` 20～40、`train.eval_period` 5～10，并检查 **val loss** 是否总体下降（详见 skill Step 7）。

---

## 变更记录（维护时可更新）

- 引入 `cfg.share.num_nodes`（`opengt/config/defaults_config.py` + `main.py`）以支持 GT-SNT 等在 `__init__` 阶段需要节点数的结构。
- 接入 **M3Dphormer**（`opengt/layer/m3dphormer_modules.py`、`opengt/network/m3dphormer.py`、`opengt/config/m3dphormer_config.py`、`configs/M3Dphormer/*.yaml`）。
