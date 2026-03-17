#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q2 – Select-All (no scoring, no weighting).

Reads every candidate file under rsr_root and writes all C×N trajectories
directly to train.jsonl.  No Q1 scores needed.

Output:
  q2_root/<student>/all/train.jsonl
  q2_root/<student>/all/train.manifest.json

Usage:
  python select_all.py \\
    --rsr_root  /home/tianruny/LIMO/data/training/rsr/RSR_data \\
    --q2_root   /home/tianruny/LIMO/data/Q2 \\
    --students  qwen2.5-7b
"""

import os
import json
import glob
import argparse
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_sua(messages: List[Dict]) -> Tuple[str, str, str]:
    sys_c = user_c = asst_c = ""
    for m in messages:
        r = m.get("role", "")
        c = m.get("content", "")
        if r == "system":      sys_c = c
        elif r == "user":      user_c = c
        elif r == "assistant": asst_c = c
    return sys_c, user_c, asst_c


def discover_candidate_files(rsr_root: str) -> List[Tuple[str, str]]:
    teacher_dirs = sorted(
        [p for p in glob.glob(os.path.join(rsr_root, "*")) if os.path.isdir(p)]
    )
    if not teacher_dirs:
        raise FileNotFoundError(f"No teacher dirs found under: {rsr_root}")
    cand_files: List[Tuple[str, str]] = []
    for td in teacher_dirs:
        teacher = os.path.basename(td)
        for fp in sorted(glob.glob(os.path.join(td, "*.json"))):
            cand_files.append((teacher, fp))
    return cand_files


def load_json_list(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def select_all(
    rsr_root: str,
    q2_root: str,
    student: str,
    n_problems: int = 5000,
):
    cand_files = discover_candidate_files(rsr_root)
    C = len(cand_files)
    print(f"[INFO] discovered {C} candidate files")

    out_dir   = os.path.join(q2_root, student, "all")
    out_path  = os.path.join(out_dir, "train.jsonl")
    ensure_dir(out_dir)

    total = 0
    manifest_files = []

    with open(out_path, "w", encoding="utf-8") as w:
        for fidx, (teacher, fp) in enumerate(cand_files):
            print(f"[LOAD] {fidx:02d} {teacher} :: {os.path.basename(fp)}")
            data = load_json_list(fp)
            n = min(n_problems, len(data))
            for pid in range(n):
                sys_c, user_c, asst_c = get_sua(data[pid]["messages"])
                obj = {
                    "messages": [
                        {"role": "system",    "content": sys_c},
                        {"role": "user",      "content": user_c},
                        {"role": "assistant", "content": asst_c},
                    ],
                    "meta": {
                        "problem_id": pid,
                        "teacher":    teacher,
                        "run_file":   os.path.basename(fp),
                        "fidx":       fidx,
                        "q2_method":  "all",
                    },
                }
                w.write(json.dumps(obj, ensure_ascii=False) + "\n")
                total += 1
            manifest_files.append({"fidx": fidx, "teacher": teacher, "path": fp, "n": n})
            del data

    manifest_path = os.path.join(out_dir, "train.manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "q2_method":   "all",
            "n_candidates": C,
            "n_problems":  n_problems,
            "total_lines": total,
            "files":       manifest_files,
        }, f, ensure_ascii=False, indent=2)

    print(f"[OK] {total} lines → {out_path}")
    print(f"[OK] {manifest_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Q2 Select-All: dump every trajectory without scoring."
    )
    ap.add_argument("--rsr_root",   type=str, required=True)
    ap.add_argument("--q2_root",    type=str, required=True)
    ap.add_argument("--students",   type=str, required=True,
                    help="Comma-separated student folder names")
    ap.add_argument("--n_problems", type=int, default=5000)
    args = ap.parse_args()

    for s in [s.strip() for s in args.students.split(",") if s.strip()]:
        print(f"\n=== student: {s} ===")
        select_all(
            rsr_root=args.rsr_root,
            q2_root=args.q2_root,
            student=s,
            n_problems=args.n_problems,
        )


if __name__ == "__main__":
    main()
