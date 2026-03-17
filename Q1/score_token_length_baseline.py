#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1 – Token Length scorer.

Assigns a score equal to the word count of the assistant text for every
(candidate, problem) pair. No GPU required.

Energy function:
  score(fidx, pid) = word_count(assistant_text)

Output:
  out_root/<student>/token_length/scores.json

scores.json schema (shared across all Q1 baselines):
  {
    "method":          "token_length",
    "n_problems":      <int>,
    "n_candidates":    <int>,
    "candidate_files": [{"fidx": 0, "teacher": "...", "path": "..."}, ...],
    "scores":          [[score_fidx0_pid0, ...], ...]   # shape [C, N]
  }

Usage:
  python score_token_length_baseline.py \\
    --rsr_root /home/tianruny/LIMO/data/training/rsr/RSR_data \\
    --out_root /home/tianruny/LIMO/data/Q1 \\
    --students qwen2.5-7b
"""

import os
import json
import glob
import argparse
import hashlib
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


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
        files   = sorted(glob.glob(os.path.join(td, "*.json")))
        if not files:
            raise FileNotFoundError(f"No *.json files under: {td}")
        for fp in files:
            cand_files.append((teacher, fp))
    return cand_files


def load_json_list(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_canonical_prompt_hashes(first_file: str, n_problems: int) -> List[str]:
    data = load_json_list(first_file)
    if len(data) < n_problems:
        raise ValueError(f"{first_file} has {len(data)} items, expected >= {n_problems}")
    hashes = []
    for i in range(n_problems):
        sys_c, user_c, _ = get_sua(data[i]["messages"])
        hashes.append(sha1_text(sys_c + "\n" + user_c))
    return hashes


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def word_count(text: str) -> int:
    return len(text.strip().split())


# ---------------------------------------------------------------------------
# Main scoring routine
# ---------------------------------------------------------------------------

def score_token_length(
    rsr_root: str,
    out_path: str,
    n_problems: int = 5000,
    verify_prompts: bool = True,
):
    method = "token_length"
    cand_files = discover_candidate_files(rsr_root)
    C = len(cand_files)
    if C < 2:
        raise RuntimeError(f"Too few candidate files discovered: {C}")
    print(f"[INFO] discovered {C} candidate files")

    canonical_hashes = None
    if verify_prompts:
        canonical_hashes = build_canonical_prompt_hashes(cand_files[0][1], n_problems)
        print("[INFO] built canonical prompt hashes from:", cand_files[0][1])

    # scores[fidx][pid] = word count of assistant text
    scores: List[List[float]] = [[0.0] * n_problems for _ in range(C)]

    for fidx, (teacher, fp) in enumerate(cand_files):
        print(f"[SCORE] {fidx:02d} {teacher} :: {os.path.basename(fp)}")
        data = load_json_list(fp)
        if len(data) < n_problems:
            raise ValueError(f"{fp} has {len(data)} items, expected >= {n_problems}")

        for pid in range(n_problems):
            ex = data[pid]
            sys_c, user_c, asst_c = get_sua(ex["messages"])

            if verify_prompts and canonical_hashes is not None:
                h = sha1_text(sys_c + "\n" + user_c)
                if h != canonical_hashes[pid]:
                    raise RuntimeError(
                        f"Prompt mismatch at pid={pid} for file={fp}\n"
                        f"Expected hash={canonical_hashes[pid]}, got hash={h}"
                    )

            scores[fidx][pid] = float(word_count(asst_c))

        del data
        print(f"  → finished {teacher}")

    # Save
    ensure_dir(os.path.dirname(out_path))
    out = {
        "method":          method,
        "n_problems":      n_problems,
        "n_candidates":    C,
        "candidate_files": [
            {"fidx": i, "teacher": t, "path": p}
            for i, (t, p) in enumerate(cand_files)
        ],
        "scores": scores,   # shape [C][n_problems], scores[fidx][pid] = float
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    print(f"[OK] wrote scores for {C} × {n_problems} = {C * n_problems} pairs")
    print(f"[OK] {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Q1 Token Length baseline: word count of assistant text per (candidate, problem)."
    )
    ap.add_argument("--rsr_root",  type=str, required=True,
                    help="Path to RSR_data root")
    ap.add_argument("--out_root",  type=str, required=True,
                    help="Q1 output root, e.g. /home/tianruny/LIMO/data/Q1")
    ap.add_argument("--students",  type=str, required=True,
                    help="Comma-separated student folder names, e.g. qwen2.5-7b,qwen3-4b")
    ap.add_argument("--n_problems", type=int, default=5000)
    ap.add_argument("--no_verify_prompts", action="store_true",
                    help="Skip cross-file prompt consistency check (faster).")
    args = ap.parse_args()

    students       = [s.strip() for s in args.students.split(",") if s.strip()]
    verify_prompts = not args.no_verify_prompts

    for s in students:
        out_dir  = os.path.join(args.out_root, s, "token_length")
        out_path = os.path.join(out_dir, "scores.json")
        print(f"\n=== student: {s} ===")
        score_token_length(
            rsr_root=args.rsr_root,
            out_path=out_path,
            n_problems=args.n_problems,
            verify_prompts=verify_prompts,
        )


if __name__ == "__main__":
    main()
