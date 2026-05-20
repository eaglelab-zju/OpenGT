#!/usr/bin/env python3
"""
从任务队列 CSV 中取一条待执行任务，调用 tune.py + Optuna 调参，并写回队列状态与结果汇总。

队列 CSV 列：model,dataset,dataset0,optuna_n_trials,status,note
  - status: pending / running / done / failed / skipped
  - dataset0: 当目标 yaml 不存在时，传给 configs_gen2.py 的模板数据集
  - optuna_n_trials: 为空则按数据集自动选择（ogbg-* / peptides-* 用 --optuna-n-trials-graph，其余用 --optuna-n-trials）

策略（默认队列生成）：仅 chameleon-new / squirrel-new；排除旧 chameleon、squirrel、zinc、
webkb-cor、wn-chameleon。失败任务同样写入结果 CSV。

默认将 **--accelerator** 传给 **tune.py --override_accelerator**，覆盖 yaml 中的设备，便于多进程
多卡并行；可用 **--disable-accelerator-override** 关闭，或用 **--train-accelerator** 单独指定训练卡。

子进程输出：**queue worker 终端 / nohup 主日志** 仅写入摘要行（如 **[Optuna] trial k/N**、设备覆盖、
trial 聚合结果）；**完整训练日志**请使用 **--task-log-dir**（否则完整输出只保留在内存 tail 供失败时写入 CSV）。
环境变量 **OPENGT_QUEUE_VERBOSE=1** 可恢复将子进程全部输出打到 worker 日志（调试用）。

后台跑与监控：见 run_queue_background.sh 末尾注释。
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _read_cfg_out_dir(cfg_yaml: Path) -> str:
    try:
        text = cfg_yaml.read_text(encoding="utf-8")
    except OSError:
        return "results"
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("out_dir:"):
            return s.split(":", 1)[1].strip().strip("\"'")
    return "results"

REPO_ROOT = Path(__file__).resolve().parent

# --write-default-queue 写入 CSV 时使用的默认 trial 数（与 argparse 默认值保持一致）
DEFAULT_OPTUNA_TRIALS_MAIN = 50
DEFAULT_OPTUNA_TRIALS_GRAPH = 15

# 与 tune.py / 各模型实现对应的关键超参说明（用于启动时打印；实际搜索在 tune.py 的 objective 中）
MODEL_KEY_HPARAMS: Dict[str, str] = {
    "Graphtransformer": "base_lr, weight_decay, gt.layers, gt.dim_hidden, gt.dropout, gt.n_heads, gt.attn_dropout",
    "GTSNT": "base_lr, weight_decay, gt.layers, gt.dim_hidden(与 n_heads 整除对齐), gt.dropout, gt.n_heads, attn_dropout, "
    "gtsnt.cb_channels, gtsnt.T, gtsnt.v_threshold, gtsnt.init_beta, gtsnt.neuron(PLIF/LIF/IF)",
    "HubGT": "base_lr, weight_decay, gt.layers, gt.dim_hidden, gt.dropout, gt.n_heads(1/2/4/8), attn_dropout, "
    "hubgt.dp_input, hubgt.ffn_ratio",
    "Polynormer": "base_lr, weight_decay, gt.layers, gt.dim_hidden, polynormer.heads, in_dropout, dropout, "
    "global_dropout, beta, global_layers, use_global",
    "CoBFormer": "base_lr, weight_decay, gt.layers, gt.dim_hidden, gt.dropout, attn_dropout, "
    "gnn.layers, gt.alpha, gt.tau",
    "NodeFormer": "base_lr, weight_decay, gt.layers, gt.dim_hidden, gt.dropout, gt.n_heads, gt.attn_dropout（见 tune.py 通用 GT 分支）",
    "M3Dphormer": "base_lr, weight_decay, gt.layers, gt.dim_hidden, gt.dropout, gt.n_heads, gt.attn_dropout（见 tune.py 通用 GT 分支）",
}

# configs_gen2.py 中显式分支支持的数据集（可与模板组合生成新 yaml）
CONFIGS_GEN2_DATASETS = frozenset(
    {
        "cora",
        "citeseer",
        "pubmed",
        "actor",
        "chameleon",
        "squirrel",
        "chameleon-new",
        "squirrel-new",
        "cornell",
        "texas",
        "wisconsin",
        "zinc",
        "ogbg-molhiv",
        "ogbg-molpcba",
        "peptides-func",
        "peptides-struct",
    }
)

# 默认任务队列中排除：旧 chameleon/squirrel、zinc、webkb-cor、wn-chameleon
EXCLUDED_QUEUE_DATASETS = frozenset(
    {"chameleon", "squirrel", "zinc", "webkb-cor", "wn-chameleon"}
)

TARGET_MODELS = (
    "Graphtransformer",
    "GTSNT",
    "HubGT",
    "Polynormer",
    "CoBFormer",
    "NodeFormer",
    "M3Dphormer",
)

RESULT_CSV_FIELDS = [
    "timestamp",
    "finished_at",
    "model",
    "dataset",
    "status",
    "optuna_n_trials",
    "wall_time_sec",
    "peak_gpu_memory_mb",
    "best_value",
    "best_trial_number",
    "study_name",
    "best_params_json",
    "metrics_json",
    "log_path",
    "stderr_tail",
    "error",
]


def _is_light_graph_dataset(ds: str) -> bool:
    return ds.startswith("ogbg-") or ds.startswith("peptides-")


def _extra_tune_args_with_default_storage(extra: List[str]) -> List[str]:
    """未显式指定 ``--optuna_storage`` 时默认 ``optuna_runs/studies_merged.db``，与 ``merge_optuna_dbs.py`` 输出一致以便恢复 study。

    多 worker 长时间并发写同一 sqlite 可能锁等待；必要时用 ``--tune-arg`` 为各进程指定独立 .db。
    """
    out = list(extra)
    for a in out:
        if a == "--optuna_storage" or a.startswith("--optuna_storage="):
            return out
    dbdir = REPO_ROOT / "optuna_runs"
    dbdir.mkdir(exist_ok=True)
    rel = (dbdir / "studies_merged.db").relative_to(REPO_ROOT)
    url = "sqlite:///" + str(rel).replace(os.sep, "/") + "?timeout=300"
    out.extend(["--optuna_storage", url])
    return out


def resolve_optuna_trials(dataset: str, cell: str, main: int, graph: int) -> int:
    """CSV 单元格有正整数则优先，否则 ogbg/peptides 用 graph，其余用 main。"""
    t = (cell or "").strip()
    if t.isdigit():
        return max(1, int(t))
    if _is_light_graph_dataset(dataset):
        return max(1, int(graph))
    return max(1, int(main))


def _yaml_path(model: str, dataset: str) -> Path:
    return REPO_ROOT / "configs" / model / f"{dataset}-{model}.yaml"


def _list_shipped_datasets(model: str) -> List[str]:
    d = REPO_ROOT / "configs" / model
    if not d.is_dir():
        return []
    out = []
    for p in d.glob(f"*-{model}.yaml"):
        stem = p.name[: -len(f"-{model}.yaml")]
        if stem and "_grid" not in stem:
            out.append(stem)
    return sorted(set(out))


def queue_dataset_universe() -> List[str]:
    """参与默认队列的数据集：configs_gen2 ∪ 各模型 shipped yaml，再减去 EXCLUDED_QUEUE_DATASETS。"""
    s = set(CONFIGS_GEN2_DATASETS)
    for m in TARGET_MODELS:
        s.update(_list_shipped_datasets(m))
    s -= EXCLUDED_QUEUE_DATASETS
    return sorted(s)


def pick_dataset0(model: str, dataset: str) -> Optional[str]:
    """为 configs_gen2 选择模板：优先目标自身，其次 cora / chameleon-new / ogbg-molhiv。"""
    if _yaml_path(model, dataset).is_file():
        return dataset
    for cand in (dataset, "cora", "chameleon-new", "ogbg-molhiv"):
        if _yaml_path(model, cand).is_file():
            return cand
    return None


def row_eligible(model: str, dataset: str) -> Tuple[bool, Optional[str]]:
    """
    若已有 yaml，或数据集在 configs_gen2 支持列表且能找到模板，则任务有效。
    返回 (eligible, dataset0)。
    """
    if _yaml_path(model, dataset).is_file():
        return True, dataset
    if dataset not in CONFIGS_GEN2_DATASETS:
        return False, None
    d0 = pick_dataset0(model, dataset)
    if d0 is None:
        return False, None
    return True, d0


def write_default_queue(path: Path) -> None:
    rows = []
    for model in TARGET_MODELS:
        for dataset in queue_dataset_universe():
            if model == "GTSNT" and _is_light_graph_dataset(dataset):
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "dataset0": "",
                        "optuna_n_trials": "",
                        "status": "skipped",
                        "note": "gtsnt_transductive_single_graph_only",
                    }
                )
                continue
            if model == "HubGT" and _is_light_graph_dataset(dataset):
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "dataset0": "",
                        "optuna_n_trials": "",
                        "status": "skipped",
                        "note": "hubgt_dense_single_graph_only",
                    }
                )
                continue
            if model == "CoBFormer" and _is_light_graph_dataset(dataset):
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "dataset0": "",
                        "optuna_n_trials": "",
                        "status": "skipped",
                        "note": "cobformer_metis_partition_ogbg",
                    }
                )
                continue
            if model == "M3Dphormer" and _is_light_graph_dataset(dataset):
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "dataset0": "",
                        "optuna_n_trials": "",
                        "status": "skipped",
                        "note": "m3dphormer_node_only_for_now",
                    }
                )
                continue
            ok, d0 = row_eligible(model, dataset)
            n_trials = (
                str(DEFAULT_OPTUNA_TRIALS_GRAPH)
                if _is_light_graph_dataset(dataset)
                else str(DEFAULT_OPTUNA_TRIALS_MAIN)
            )
            if not ok:
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "dataset0": "",
                        "optuna_n_trials": "",
                        "status": "skipped",
                        "note": "no_template_or_unsupported_by_configs_gen2",
                    }
                )
                continue
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "dataset0": d0,
                    "optuna_n_trials": n_trials,
                    "status": "pending",
                    "note": "",
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "dataset", "dataset0", "optuna_n_trials", "status", "note"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"已写入默认队列: {path} ，共 {len(rows)} 行（含 skipped）。")


def read_queue(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        return [], ["model", "dataset", "dataset0", "optuna_n_trials", "status", "note"]
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    fieldnames = reader.fieldnames or []
    return list(reader), fieldnames


def write_queue_atomic(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    base_fields = ["model", "dataset", "dataset0", "optuna_n_trials", "status", "note"]
    extra = [c for c in fieldnames if c not in base_fields]
    out_fields = base_fields + extra
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = {k: r.get(k, "") for k in out_fields}
            w.writerow(row)
    tmp.replace(path)


def lock_file(fp) -> None:
    fcntl.flock(fp.fileno(), fcntl.LOCK_EX)


def unlock_file(fp) -> None:
    fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


def ensure_config(
    model: str,
    dataset: str,
    dataset0: str,
    accelerator: str,
) -> Tuple[bool, str]:
    target = _yaml_path(model, dataset)
    if target.is_file():
        return True, str(target)
    if not dataset0:
        return False, "缺少 dataset0，无法调用 configs_gen2 生成配置"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "configs_gen2.py"),
        "--model",
        model,
        "--dataset",
        dataset,
        "--dataset0",
        dataset0,
        "--accelerator",
        accelerator,
    ]
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        return False, (r.stderr or r.stdout or "configs_gen2 失败")[:4000]
    if not target.is_file():
        return False, "configs_gen2 未生成预期 yaml"
    return True, str(target)


def read_optuna_best(cfg_yaml: Path) -> Optional[dict]:
    out_dir = _read_cfg_out_dir(cfg_yaml)
    run_name = cfg_yaml.stem
    json_path = (REPO_ROOT / out_dir / run_name / "optuna_best.json").resolve()
    if not json_path.is_file():
        return None
    try:
        with json_path.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def collect_best_trial_metrics_and_peak(
    cfg_yaml: Path,
    best_summary: Optional[dict],
    job_wall_sec: float,
) -> Tuple[str, str, str]:
    """
    读取最优 trial 目录下的 runtime_stats.json（显存）与 agg/{train,val,test}/best.json。
    返回 (peak_gpu_memory_mb 字符串, metrics_json, best_trial_number 字符串)。
    """
    bundle: dict = {
        "job_wall_time_sec": round(job_wall_sec, 3),
        "train": {},
        "val": {},
        "test": {},
    }
    if best_summary:
        bundle["optuna"] = {
            k: best_summary.get(k)
            for k in ("best_value", "study_name", "model_type", "dataset", "layer_type")
        }
        if "best_params" in best_summary:
            bundle["optuna"]["best_params"] = best_summary["best_params"]

    if not best_summary or "best_trial_number" not in best_summary:
        return "", json.dumps(bundle, ensure_ascii=False, default=str), ""

    tnum = int(best_summary["best_trial_number"])
    run_name = cfg_yaml.stem
    out_root = _read_cfg_out_dir(cfg_yaml)
    trial_dir = (REPO_ROOT / out_root / run_name / f"trial_{tnum}").resolve()

    peak_str = ""
    rs = trial_dir / "runtime_stats.json"
    if rs.is_file():
        try:
            with rs.open(encoding="utf-8") as f:
                data = json.load(f)
            v = data.get("peak_cuda_memory_mb")
            if v is not None:
                peak_str = str(round(float(v), 6))
        except (json.JSONDecodeError, ValueError, TypeError, OSError):
            pass

    for split in ("train", "val", "test"):
        p = trial_dir / "agg" / split / "best.json"
        if not p.is_file():
            continue
        try:
            with p.open(encoding="utf-8") as f:
                bundle[split] = json.load(f)
        except (json.JSONDecodeError, OSError):
            bundle[split] = {"_read_error": str(p)}

    return peak_str, json.dumps(bundle, ensure_ascii=False, default=str), str(tnum)


def _compact_failure_reason(log_tail: str, code: int) -> str:
    """从子进程输出尾部提取简短失败标签，禁止把整段 traceback 写入 CSV。"""
    t = (log_tail or "")[-16000:]
    if "CUDA out of memory" in t or "OutOfMemoryError" in t:
        tag = "cuda_oom"
    elif "NotImplementedError" in t and "HubGT" in t and "batched graphs" in t:
        tag = "hubgt_batched_graph_unsupported"
    elif "sqlite3.OperationalError" in t and "database is locked" in t:
        tag = "sqlite_locked"
    elif "Sizes of tensors must match" in t and "collate" in t:
        tag = "dataset_collate_shape"
    elif "KeyboardInterrupt" in t:
        tag = "interrupted"
    elif "UnpicklingError" in t or "Weights only load failed" in t:
        tag = "torch_load_unpickle"
    elif "mat1 and mat2 shapes cannot be multiplied" in t:
        tag = "linear_shape_mismatch"
    else:
        tag = "tune_failed"
    return f"{tag}|exit={int(code)}"


def append_result_csv(
    results_csv: Path,
    fields: Dict[str, str],
) -> None:
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = results_csv.is_file()
    row = {k: fields.get(k, "") for k in RESULT_CSV_FIELDS}
    with results_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_CSV_FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


def _queue_worker_summary_line(line: str) -> bool:
    """是否将本行写入 worker 的 stdout（nohup 主日志）。默认只放行摘要，避免训练日志刷屏。"""
    if os.environ.get("OPENGT_QUEUE_VERBOSE", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    if line.startswith("Traceback"):
        return True
    return any(
        m in line
        for m in (
            "[Optuna]",
            "[tune] override_accelerator",
            "Final result for this trial:",
        )
    )


def _run_tune_streaming(
    cmd: List[str],
    cwd: str,
    log_path: Optional[Path],
) -> Tuple[int, str]:
    """运行 tune.py：完整输出写入 log_path（若给定）；worker stdout 仅写摘要行。返回 (exit_code, tail)。"""
    log_f = None
    tail_chunks: List[str] = []
    # 仅用于无 task 日志时拼短摘要；勿攒大块 traceback 进内存（也不写入 results CSV）。
    max_tail_chars = 1200
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_f = log_path.open("w", encoding="utf-8")
        p = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            if log_f is not None:
                log_f.write(line)
                log_f.flush()
            if _queue_worker_summary_line(line):
                sys.stdout.write(line)
                sys.stdout.flush()
            tail_chunks.append(line)
            joined = "".join(tail_chunks)
            if len(joined) > max_tail_chars:
                tail_chunks = [joined[-max_tail_chars:]]
        p.stdout.close()
        rc = int(p.wait())
        tail = "".join(tail_chunks)
        if len(tail) > max_tail_chars:
            tail = tail[-max_tail_chars:]
        return rc, tail
    finally:
        if log_f is not None:
            log_f.close()


def run_one_task(
    cfg_path: Path,
    optuna_n_trials: int,
    repeat: int,
    extra_tune_args: List[str],
    task_log: Optional[Path],
    override_accelerator: str,
) -> Tuple[int, str, str, str, float]:
    cmd: List[str] = [
        sys.executable,
        str(REPO_ROOT / "tune.py"),
        "--optuna_n_trials",
        str(optuna_n_trials),
        "--repeat",
        str(repeat),
        "--cfg",
        str(cfg_path.resolve().relative_to(REPO_ROOT.resolve())),
    ] + _extra_tune_args_with_default_storage(list(extra_tune_args))
    oa = (override_accelerator or "").strip()
    if oa:
        cmd.extend(["--override_accelerator", oa])
    cmd_str = " ".join(cmd)
    print("[运行]", cmd_str, flush=True)
    log_path_str = str(task_log.resolve()) if task_log is not None else ""
    t0 = time.perf_counter()
    code, tail = _run_tune_streaming(cmd, str(REPO_ROOT), task_log)
    wall = time.perf_counter() - t0
    return code, cmd_str, log_path_str, tail, wall


def find_next_pending(rows: List[Dict[str, str]]) -> int:
    for i, r in enumerate(rows):
        st = (r.get("status") or "").strip().lower()
        if st in ("", "pending"):
            return i
    return -1


def study_name_for_cfg_yaml(cfg_yaml: Path) -> str:
    """与 tune.py 中 Optuna study_name 规则一致（无 study_suffix）。"""
    import sys

    import opengt  # noqa: F401, register
    from torch_geometric.graphgym.cmd_args import parse_args
    from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg

    cfg_arg = str(cfg_yaml.resolve().relative_to(REPO_ROOT.resolve()))
    sys.argv = ["queue_audit_study", "--cfg", cfg_arg, "--repeat", "1"]
    set_cfg(cfg)
    load_cfg(cfg, parse_args())
    node_encoder_name = cfg.dataset.node_encoder_name
    if not cfg.dataset.node_encoder:
        node_encoder_name = "none"
    model_name = cfg.model.type
    if cfg.dataset.node_encoder is True:
        model_name += "+" + node_encoder_name
    if cfg.model.type != "Graphormer":
        model_name += "+" + cfg.gt.layer_type
    layer_name = getattr(cfg.gt, "layer_type", "none")
    return f"my_study_{model_name}_{cfg.dataset.name}_{layer_name}"


def audit_optuna_db_vs_tasks(
    db_path: Path,
    queue_path: Path,
    optuna_nt_main: int,
    optuna_nt_graph: int,
) -> None:
    """
    对照 experiment_tasks：根据各任务 yaml 推导的 study 名，在指定 Optuna SQLite 中统计
    ``len(study.trials)``；若少于 CSV 要求的 optuna_n_trials，则将 ``done`` / ``running`` 改为 ``pending``。
    """
    import optuna

    if not db_path.is_file():
        print(f"[audit] 数据库不存在，跳过: {db_path}", flush=True)
        return

    storage = "sqlite:///" + str(db_path.resolve()).replace(os.sep, "/")
    try:
        summaries = optuna.study.get_all_study_summaries(storage=storage)
    except Exception as e:
        print(f"[audit] 无法读取 Optuna 存储: {e}", flush=True)
        return

    name_to_n = {s.study_name: int(s.n_trials) for s in summaries}
    print(f"[audit] {db_path} 中共 {len(name_to_n)} 个 study。", flush=True)

    rows, fieldnames = read_queue(queue_path)
    changed = 0
    for row in rows:
        st = (row.get("status") or "").strip().lower()
        if st in ("skipped", "pending", "failed"):
            continue
        if st not in ("done", "running"):
            continue
        model = (row.get("model") or "").strip()
        dataset = (row.get("dataset") or "").strip()
        cfg_yaml = _yaml_path(model, dataset)
        if not cfg_yaml.is_file():
            continue
        try:
            sn = study_name_for_cfg_yaml(cfg_yaml)
        except Exception as e:
            print(f"[audit] 无法解析 {cfg_yaml}: {e}", flush=True)
            continue
        n_goal = resolve_optuna_trials(
            dataset,
            row.get("optuna_n_trials", "") or "",
            optuna_nt_main,
            optuna_nt_graph,
        )
        n_db = name_to_n.get(sn)
        if n_db is None:
            if st == "running":
                row["status"] = "pending"
                row["note"] = "audit_no_matching_study_in_db"
                changed += 1
            continue
        if n_db < n_goal:
            row["status"] = "pending"
            row["note"] = f"audit_optuna_trials_{n_db}_lt_{n_goal}"
            changed += 1
            print(
                f"[audit] 重置: {model} {dataset} study={sn!r} trials={n_db} < {n_goal}",
                flush=True,
            )

    if changed:
        write_queue_atomic(queue_path, rows, fieldnames or [])
    print(f"[audit] 已更新 {changed} 条任务为 pending。", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenGT_eval 队列化 Optuna 实验")
    ap.add_argument(
        "--queue",
        type=Path,
        default=REPO_ROOT / "experiment_tasks.csv",
        help="任务队列 CSV",
    )
    ap.add_argument(
        "--results-csv",
        type=Path,
        default=REPO_ROOT / "experiment_results.csv",
        help="结果追加写入的 CSV",
    )
    ap.add_argument("--optuna-n-trials", type=int, default=DEFAULT_OPTUNA_TRIALS_MAIN)
    ap.add_argument(
        "--optuna-n-trials-graph",
        type=int,
        default=DEFAULT_OPTUNA_TRIALS_GRAPH,
        help="ogbg-* 与 peptides-* 在 CSV 的 optuna_n_trials 为空时使用的默认 trial 数",
    )
    ap.add_argument("--repeat", type=int, default=3, help="传给 tune.py 的 --repeat（多种子）")
    ap.add_argument("--accelerator", type=str, default="cuda:0", help="生成缺失 yaml 时传给 configs_gen2")
    ap.add_argument(
        "--task-log-dir",
        type=Path,
        default=None,
        help="每个任务完整 tune 日志目录；若不指定，完整输出不会写入磁盘（worker 日志仅摘要）",
    )
    ap.add_argument(
        "--until-empty",
        action="store_true",
        help="连续处理队列中所有 pending 任务直到没有 pending",
    )
    ap.add_argument(
        "--write-default-queue",
        action="store_true",
        help="根据内置模型与数据集并集重写 --queue（慎用覆盖）",
    )
    ap.add_argument(
        "--tune-arg",
        action="append",
        default=[],
        help="可重复指定，原样追加传给 tune.py（例：--tune-arg=--optuna_storage --tune-arg=sqlite:///study.db）",
    )
    ap.add_argument(
        "--train-accelerator",
        type=str,
        default=None,
        help="传给 tune.py --override_accelerator；默认与 --accelerator 相同",
    )
    ap.add_argument(
        "--disable-accelerator-override",
        action="store_true",
        help="不向 tune.py 传 --override_accelerator，完全使用 yaml 中的 accelerator",
    )
    ap.add_argument(
        "--audit-optuna-db",
        type=Path,
        nargs="?",
        const=REPO_ROOT / "my_study.db",
        default=None,
        help=(
            "仅审计：读取该 SQLite 中各 study 的 trial 数，与 experiment_tasks.csv 中 optuna_n_trials 对照；"
            "不足则将 done/running 改为 pending 后退出（不写 experiment_results）。"
        ),
    )
    args = ap.parse_args()
    if getattr(args, "audit_optuna_db", None) is not None:
        audit_optuna_db_vs_tasks(
            args.audit_optuna_db.resolve(),
            args.queue.resolve(),
            args.optuna_n_trials,
            args.optuna_n_trials_graph,
        )
        return

    if args.disable_accelerator_override:
        train_oa = ""
    else:
        train_oa = (args.train_accelerator or args.accelerator or "").strip()

    if args.write_default_queue:
        write_default_queue(args.queue.resolve())
        if not args.until_empty:
            return

    queue_path = args.queue.resolve()
    if not queue_path.is_file():
        print(f"队列文件不存在: {queue_path} ，可先运行 --write-default-queue", file=sys.stderr)
        sys.exit(1)

    while True:
        lock_path = queue_path.with_name(queue_path.name + ".lock")
        with lock_path.open("w", encoding="utf-8") as lf:
            lock_file(lf)
            try:
                rows, fieldnames = read_queue(queue_path)
                idx = find_next_pending(rows)
                if idx < 0:
                    print("队列中没有 pending 任务。", flush=True)
                    break

                row = rows[idx]
                model = (row.get("model") or "").strip()
                dataset = (row.get("dataset") or "").strip()
                dataset0 = (row.get("dataset0") or "").strip()
                nt_cell = row.get("optuna_n_trials", "") or ""

                rows[idx]["status"] = "running"
                rows[idx]["note"] = f"started_ts={time.time():.0f}"
                write_queue_atomic(queue_path, rows, fieldnames or [])
            finally:
                unlock_file(lf)

        n_trials = resolve_optuna_trials(
            dataset, nt_cell, args.optuna_n_trials, args.optuna_n_trials_graph
        )
        task_log: Optional[Path] = None
        if args.task_log_dir is not None:
            tdir = args.task_log_dir.resolve()
            safe_m = model.replace("/", "_").replace(os.sep, "_")
            safe_d = dataset.replace("/", "_").replace(os.sep, "_")
            task_log = tdir / f"{safe_m}__{safe_d}__{int(time.time())}.log"

        print(
            f"\n======== 任务: model={model} dataset={dataset} dataset0={dataset0} "
            f"optuna_n_trials={n_trials} ========",
            flush=True,
        )
        desc = MODEL_KEY_HPARAMS.get(model, "见 tune.py objective 通用分支")
        print(f"本模型关键调参维度: {desc}", flush=True)

        ok_cfg, msg = ensure_config(model, dataset, dataset0, args.accelerator)
        cfg_p = Path(msg) if ok_cfg else None
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        if not ok_cfg:
            lock_path = queue_path.with_name(queue_path.name + ".lock")
            with lock_path.open("w", encoding="utf-8") as lf:
                lock_file(lf)
                try:
                    rows, fieldnames = read_queue(queue_path)
                    for i, r in enumerate(rows):
                        if (
                            r.get("model") == model
                            and r.get("dataset") == dataset
                            and r.get("status") == "running"
                        ):
                            rows[i]["status"] = "failed"
                            rows[i]["note"] = msg[:500]
                            break
                    write_queue_atomic(queue_path, rows, fieldnames or [])
                finally:
                    unlock_file(lf)
            ts_end = time.strftime("%Y-%m-%d %H:%M:%S")
            append_result_csv(
                args.results_csv.resolve(),
                {
                    "timestamp": ts,
                    "finished_at": ts_end,
                    "model": model,
                    "dataset": dataset,
                    "status": "failed",
                    "optuna_n_trials": str(n_trials),
                    "wall_time_sec": "0",
                    "peak_gpu_memory_mb": "",
                    "best_value": "",
                    "best_trial_number": "",
                    "study_name": "",
                    "best_params_json": "",
                    "metrics_json": json.dumps(
                        {"phase": "config", "detail": (msg or "")[:400]},
                        ensure_ascii=False,
                    ),
                    "log_path": "",
                    "stderr_tail": "",
                    "error": "config_or_gen_failed",
                },
            )
        else:
            code, cmd_s, log_path_str, log_tail, wall_sec = run_one_task(
                cfg_p,
                n_trials,
                args.repeat,
                list(args.tune_arg),
                task_log,
                train_oa,
            )
            ts_end = time.strftime("%Y-%m-%d %H:%M:%S")
            best = read_optuna_best(cfg_p)
            peak_mb, metrics_json, btnum = collect_best_trial_metrics_and_peak(
                cfg_p, best, wall_sec
            )
            lock_path = queue_path.with_name(queue_path.name + ".lock")
            with lock_path.open("w", encoding="utf-8") as lf:
                lock_file(lf)
                try:
                    rows, fieldnames = read_queue(queue_path)
                    for i, r in enumerate(rows):
                        if (
                            r.get("model") == model
                            and r.get("dataset") == dataset
                            and r.get("status") == "running"
                        ):
                            rows[i]["status"] = "done" if code == 0 else "failed"
                            rows[i]["note"] = cmd_s if code != 0 else "ok"
                            break
                    write_queue_atomic(queue_path, rows, fieldnames or [])
                finally:
                    unlock_file(lf)

            if code == 0:
                append_result_csv(
                    args.results_csv.resolve(),
                    {
                        "timestamp": ts,
                        "finished_at": ts_end,
                        "model": model,
                        "dataset": dataset,
                        "status": "done",
                        "optuna_n_trials": str(n_trials),
                        "wall_time_sec": str(round(wall_sec, 3)),
                        "peak_gpu_memory_mb": peak_mb,
                        "best_value": ""
                        if best is None
                        else str(best.get("best_value", "")),
                        "best_trial_number": btnum,
                        "study_name": ""
                        if best is None
                        else str(best.get("study_name", "")),
                        "best_params_json": ""
                        if best is None
                        else json.dumps(
                            best.get("best_params"), ensure_ascii=False
                        ),
                        "metrics_json": metrics_json,
                        "log_path": log_path_str,
                        "stderr_tail": "",
                        "error": "",
                    },
                )
            else:
                reason = _compact_failure_reason(log_tail, code)
                append_result_csv(
                    args.results_csv.resolve(),
                    {
                        "timestamp": ts,
                        "finished_at": ts_end,
                        "model": model,
                        "dataset": dataset,
                        "status": "failed",
                        "optuna_n_trials": str(n_trials),
                        "wall_time_sec": str(round(wall_sec, 3)),
                        "peak_gpu_memory_mb": peak_mb,
                        "best_value": "",
                        "best_trial_number": btnum,
                        "study_name": ""
                        if best is None
                        else str(best.get("study_name", "")),
                        "best_params_json": "",
                        "metrics_json": json.dumps(
                            {
                                "phase": "tune",
                                "reason": reason.split("|", 1)[0],
                                "log_path": log_path_str,
                            },
                            ensure_ascii=False,
                        ),
                        "log_path": log_path_str,
                        "stderr_tail": "",
                        "error": reason[:200],
                    },
                )

        if not args.until_empty:
            break


if __name__ == "__main__":
    main()
