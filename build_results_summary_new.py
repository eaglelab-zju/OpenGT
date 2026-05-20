#!/usr/bin/env python3
"""
从 experiment_results.csv（及可选的 experiment_tasks.csv）汇总各 (model, dataset)
的指标与方差列（*_std），默认使用 Optuna 最优 trial 对应 split 的聚合结果。

数据来源优先级：
1. ``metrics_json`` 中的 train / val / test 字典（与队列写入格式一致）
2. 若 test 为空但存在 ``log_path``、``best_trial_number`` 与本地 ``configs/<model>/<dataset>-<model>.yaml``，
   则尝试读取 ``<out_dir>/<run>/trial_<n>/agg/<split>/best.json`` 回填

输出：默认 ``results_summary_new.csv``（宽表：每个指标一列 + *_std 方差列）。

用法::

    cd OpenGT_eval
    python build_results_summary_new.py
    python build_results_summary_new.py --split test --results experiment_results.csv
    python build_results_summary_new.py --split test val --out my_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent

SUMMARY_CORE_FIELDS = [
    "model",
    "dataset",
    "result_status",
    "task_status",
    "task_note",
    "task_optuna_n_trials",
    "timestamp",
    "finished_at",
    "best_value",
    "best_trial_number",
    "optuna_n_trials",
    "study_name",
    "wall_time_sec",
    "peak_gpu_memory_mb",
    "log_path",
    "error",
]

SKIP_EXACT = frozenset(
    {
        "epoch",
        "lr",
        "lr_std",
        "params",
        "params_std",
    }
)
SKIP_PREFIXES = (
    "time_",
    "eta",
    "preprocess_",
    "avg_epoch_",
    "peak_",
    "training_time_",
)


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


def _fmt_scalar(v: Any) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        if isinstance(v, float):
            return f"{v:.8g}".rstrip("0").rstrip(".")
        return str(v)
    return str(v)


def _is_metric_key(k: str) -> bool:
    if k.endswith("_std"):
        return False
    if k in SKIP_EXACT:
        return False
    if any(k.startswith(p) for p in SKIP_PREFIXES):
        return False
    return True


def _flatten_split(split_name: str, d: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        if not _is_metric_key(k):
            continue
        col = f"{split_name}_{k}"
        if isinstance(v, (int, float, str)) or v is None:
            out[col] = _fmt_scalar(v)
        else:
            out[col] = json.dumps(v, ensure_ascii=False)
        sk = f"{k}_std"
        if sk in d:
            out[f"{split_name}_{k}_std"] = _fmt_scalar(d.get(sk))
    return out


def _parse_ts(s: str) -> Tuple[int, str]:
    """用于排序：可解析则返回 (0, iso)，否则 (1, 原串)。"""
    s = (s or "").strip()
    if not s:
        return (1, "")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            from datetime import datetime

            t = datetime.strptime(s[:19], fmt)
            return (0, t.isoformat())
        except ValueError:
            continue
    return (1, s)


def _load_json_dict(raw: str) -> dict:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except json.JSONDecodeError:
        return {}


def _recover_split_from_disk(
    model: str,
    dataset: str,
    best_trial_number: str,
    split: str,
) -> dict:
    cfg = REPO_ROOT / "configs" / model / f"{dataset}-{model}.yaml"
    if not cfg.is_file():
        return {}
    out_dir = _read_cfg_out_dir(cfg)
    run_name = cfg.stem
    tnum = (best_trial_number or "").strip()
    if not tnum.isdigit():
        ob = REPO_ROOT / out_dir / run_name / "optuna_best.json"
        if ob.is_file():
            try:
                b = json.loads(ob.read_text(encoding="utf-8"))
                tnum = str(b.get("best_trial_number", "")).strip()
            except (json.JSONDecodeError, OSError):
                tnum = ""
    if not tnum.isdigit():
        return {}
    p = REPO_ROOT / out_dir / run_name / f"trial_{int(tnum)}" / "agg" / split / "best.json"
    if not p.is_file():
        return {}
    try:
        o = json.loads(p.read_text(encoding="utf-8"))
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _merge_bundle_from_row(row: Dict[str, str], splits: Tuple[str, ...]) -> dict:
    bundle = _load_json_dict(row.get("metrics_json", ""))
    bt = (row.get("best_trial_number") or "").strip()
    model = (row.get("model") or "").strip()
    dataset = (row.get("dataset") or "").strip()
    for sp in splits:
        cur = bundle.get(sp)
        if isinstance(cur, dict) and cur:
            continue
        recovered = _recover_split_from_disk(model, dataset, bt, sp)
        if recovered:
            bundle[sp] = recovered
    return bundle


def _read_tasks_csv(path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    if not path.is_file():
        return {}
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            m = (row.get("model") or "").strip()
            d = (row.get("dataset") or "").strip()
            if not m or not d:
                continue
            out[(m, d)] = {
                "task_status": (row.get("status") or "").strip(),
                "task_note": (row.get("note") or "").strip(),
                "task_optuna_n_trials": (row.get("optuna_n_trials") or "").strip(),
            }
    return out


def _read_results_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        print(f"[error] 找不到结果文件: {path}", file=sys.stderr)
        sys.exit(1)
    rows: List[Dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
    return rows


def _pick_latest_per_pair(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    best: Dict[Tuple[str, str], Tuple[Tuple[int, str], Dict[str, str]]] = {}
    for row in rows:
        m = (row.get("model") or "").strip()
        d = (row.get("dataset") or "").strip()
        if not m or not d:
            continue
        key = (m, d)
        ts = _parse_ts(row.get("finished_at", "") or row.get("timestamp", ""))
        prev = best.get(key)
        if prev is None or ts > prev[0]:
            best[key] = (ts, row)
    out = [v[1] for _, v in sorted(best.items(), key=lambda kv: (kv[0][0], kv[0][1]))]
    return out


def build_rows(
    rows: List[Dict[str, str]],
    tasks: Dict[Tuple[str, str], Dict[str, str]],
    splits: Tuple[str, ...],
) -> Tuple[List[Dict[str, str]], List[str]]:
    metric_cols: Set[str] = set()
    out_rows: List[Dict[str, str]] = []
    for row in rows:
        m = (row.get("model") or "").strip()
        d = (row.get("dataset") or "").strip()
        bundle = _merge_bundle_from_row(row, splits)
        flat: Dict[str, str] = {
            "model": m,
            "dataset": d,
            "result_status": (row.get("status") or "").strip(),
            "timestamp": (row.get("timestamp") or "").strip(),
            "finished_at": (row.get("finished_at") or "").strip(),
            "best_value": (row.get("best_value") or "").strip(),
            "best_trial_number": (row.get("best_trial_number") or "").strip(),
            "optuna_n_trials": (row.get("optuna_n_trials") or "").strip(),
            "study_name": (row.get("study_name") or "").strip(),
            "wall_time_sec": (row.get("wall_time_sec") or "").strip(),
            "peak_gpu_memory_mb": (row.get("peak_gpu_memory_mb") or "").strip(),
            "log_path": (row.get("log_path") or "").strip(),
            "error": (row.get("error") or "").strip(),
        }
        tinfo = tasks.get((m, d), {})
        flat["task_status"] = tinfo.get("task_status", "")
        flat["task_note"] = tinfo.get("task_note", "")
        flat["task_optuna_n_trials"] = tinfo.get("task_optuna_n_trials", "")

        for sp in splits:
            part = _flatten_split(sp, bundle.get(sp))
            metric_cols.update(part.keys())
            flat.update(part)

        out_rows.append(flat)

    core = SUMMARY_CORE_FIELDS
    rest: List[str] = []
    for sp in splits:
        rest.extend(sorted(c for c in metric_cols if c.startswith(f"{sp}_")))
    header = core + rest
    return out_rows, header


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="从 experiment_results 生成 results_summary_new.csv")
    ap.add_argument(
        "--results",
        type=Path,
        default=REPO_ROOT / "experiment_results.csv",
        help="队列追加的实验结果 CSV",
    )
    ap.add_argument(
        "--tasks",
        type=Path,
        default=REPO_ROOT / "experiment_tasks.csv",
        help="可选：合并任务状态/备注",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results_summary_new.csv",
        help="输出宽表路径",
    )
    ap.add_argument(
        "--split",
        nargs="+",
        default=["test"],
        choices=("train", "val", "test"),
        help="写入哪些数据划分（列名前缀 train_/val_/test_）",
    )
    ap.add_argument(
        "--no-dedupe-latest",
        action="store_true",
        help="默认每个 (model,dataset) 只保留 finished_at 最新的一条；加此选项则输出全部结果行",
    )
    ap.add_argument(
        "--no-tasks",
        action="store_true",
        help="不读取 experiment_tasks.csv",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    splits = tuple(args.split)
    all_rows = _read_results_csv(args.results.resolve())
    rows = all_rows if args.no_dedupe_latest else _pick_latest_per_pair(all_rows)
    tasks: Dict[Tuple[str, str], Dict[str, str]] = {}
    if not args.no_tasks:
        tasks = _read_tasks_csv(args.tasks.resolve())

    out_rows, header = build_rows(rows, tasks, splits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in header})

    n_core = len(SUMMARY_CORE_FIELDS)
    print(
        f"[done] 写入 {args.out} ，行数={len(out_rows)} ，指标列数={len(header) - n_core}",
        flush=True,
    )


if __name__ == "__main__":
    main()
