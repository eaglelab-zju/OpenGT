#!/usr/bin/env python3
"""
将多个 Optuna SQLite 库合并到一个库（不删除、不修改源 .db 文件）。

同名 study：把各源库中的 trial 通过 ``Study.add_trials`` 依次并入目标库中的同一
study（trial 编号由 Optuna 重新分配，与 ``optuna.copy_study`` 行为一致）。

合并后请使用同一 ``--optuna_storage`` URL 运行 ``tune.py``（仓库默认已指向
``optuna_runs/studies_merged.db``），以便 ``load_if_exists=True`` 在中断后继续跑满
``n_trials``。

示例::

    python merge_optuna_dbs.py --overwrite-merged
    python merge_optuna_dbs.py --scan-root optuna_runs --overwrite-merged
    python merge_optuna_dbs.py --list

    # 仅查看某 study 进度（任意 sqlite 库）
    python merge_optuna_dbs.py --storage sqlite:///optuna_runs/studies_merged.db --study-name 'my_study_...'
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent


def _sqlite_url(path: Path) -> str:
    return "sqlite:///" + str(path.resolve()).replace(os.sep, "/") + "?timeout=300"


def _discover_db_files(roots: Iterable[Path], exclude: Path) -> List[Path]:
    out: List[Path] = []
    ex = exclude.resolve()
    for root in roots:
        r = root.resolve()
        if not r.is_dir():
            continue
        for p in r.rglob("*.db"):
            if p.resolve() == ex:
                continue
            name = p.name
            if name.endswith("-journal") or name.endswith(".db-journal"):
                continue
            if not p.is_file():
                continue
            out.append(p)
    return sorted(set(out), key=lambda x: str(x))


def _summaries_safe(storage: str):
    import optuna

    try:
        return optuna.study.get_all_study_summaries(storage=storage)
    except Exception as e:
        print(f"[skip] 无法读取 {storage}: {e}", file=sys.stderr)
        return []


def merge_studies(
    source_dbs: List[Path],
    out_db: Path,
    overwrite_merged: bool,
) -> None:
    import optuna

    out_db.parent.mkdir(parents=True, exist_ok=True)
    merged = _sqlite_url(out_db)
    if out_db.is_file() and overwrite_merged:
        out_db.unlink()
        print(f"[info] 已删除旧合并库: {out_db}")
    elif out_db.is_file() and not overwrite_merged:
        print(
            "[warn] 输出库已存在：将把各源库 trial 追加写入；若源集合未变，可能产生重复 trial。"
            " 需要从零重建请使用 --overwrite-merged。",
            file=sys.stderr,
        )

    # study_name -> ordered list of source sqlite urls
    by_name: Dict[str, List[str]] = defaultdict(list)
    seen_pair: set[Tuple[str, str]] = set()
    for db in source_dbs:
        url = _sqlite_url(db)
        for s in _summaries_safe(url):
            key = (s.study_name, url)
            if key in seen_pair:
                continue
            seen_pair.add(key)
            by_name[s.study_name].append(url)

    n_studies = 0
    n_trials_added = 0
    for name in sorted(by_name.keys()):
        urls = by_name[name]
        merged_study = None
        for src_url in urls:
            try:
                src = optuna.load_study(study_name=name, storage=src_url)
            except Exception as e:
                print(f"[warn] 跳过 {name} @ {src_url}: {e}", file=sys.stderr)
                continue
            trials = src.get_trials(deepcopy=True)
            if merged_study is None:
                merged_study = optuna.create_study(
                    study_name=name,
                    storage=merged,
                    directions=src.directions,
                    load_if_exists=True,
                )
            else:
                merged_study = optuna.load_study(study_name=name, storage=merged)
            if not trials:
                continue
            merged_study.add_trials(trials)
            n_trials_added += len(trials)
            merged_study = optuna.load_study(study_name=name, storage=merged)
            print(
                f"[merge] {name!r} <- {src_url}  (+{len(trials)} trials, "
                f"dst_total={len(merged_study.trials)})"
            )
        if merged_study is not None:
            n_studies += 1

    print(
        f"[done] 合并库: {out_db} ；studies={n_studies} ，累计写入 trial 行数={n_trials_added}"
    )


def cmd_list(storage: str) -> None:
    import optuna

    for s in optuna.study.get_all_study_summaries(storage=storage):
        print(f"{s.study_name}\tn_trials={s.n_trials}")


def cmd_study_detail(storage: str, study_name: str) -> None:
    import optuna
    from optuna.trial import TrialState

    st = optuna.load_study(study_name=study_name, storage=storage)
    complete = sum(1 for t in st.trials if t.state == TrialState.COMPLETE)
    print(f"study={study_name}")
    print(f"  trials_total={len(st.trials)}  complete={complete}")
    if st.trials:
        last = st.trials[-1]
        print(f"  last_trial_number={last.number}  state={last.state}")


def main() -> None:
    ap = argparse.ArgumentParser(description="合并 Optuna SQLite 存储（同名 study 合并 trial）")
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "optuna_runs" / "studies_merged.db",
        help="输出合并库路径（默认 optuna_runs/studies_merged.db）",
    )
    ap.add_argument(
        "--scan-root",
        type=Path,
        action="append",
        default=[],
        help="递归扫描 *.db 的根目录，可重复。若给定则只扫描这些目录；未给定时默认扫描本脚本所在仓库根目录。",
    )
    ap.add_argument(
        "--overwrite-merged",
        action="store_true",
        help="若输出文件已存在则先删除再合并（只删合并目标，不删源库）",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="列出 --storage 中所有 study 及 n_trials 后退出",
    )
    ap.add_argument(
        "--storage",
        type=str,
        default="",
        help="与 --list / --study-name 配合的 storage URL",
    )
    ap.add_argument(
        "--study-name",
        type=str,
        default="",
        help="打印指定 study 的 trial 统计后退出",
    )
    args = ap.parse_args()

    if args.scan_root:
        roots = [Path(p) for p in args.scan_root]
    else:
        roots = [REPO_ROOT]
    out = args.out.resolve()

    if args.list:
        if not args.storage:
            args.storage = _sqlite_url(out)
        cmd_list(args.storage)
        return
    if args.study_name:
        if not args.storage:
            args.storage = _sqlite_url(out)
        cmd_study_detail(args.storage, args.study_name)
        return

    sources = _discover_db_files(roots, exclude=out)
    if not sources:
        print("[warn] 未发现任何源 .db 文件", file=sys.stderr)
        sys.exit(1)
    print(f"[info] 发现 {len(sources)} 个源库文件")
    merge_studies(sources, out, overwrite_merged=bool(args.overwrite_merged))


if __name__ == "__main__":
    main()
