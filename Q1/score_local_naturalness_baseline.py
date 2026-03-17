#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1 – Local Naturalness scorer.

Computes the step-level local naturalness score for every (candidate, problem)
pair and writes the full C×N score matrix to scores.json.

Energy function (paper Appendix C.1):
  score_local(τ) = (1/J) Σ_{j=1}^{J} [
      (1/|s_j|) Σ_k log p_θ(s_{j,k} | x, s_{j-m:j-1}, s_{j,<k})
  ]
  where m = local context window size (default 4).

Output:
  out_root/<student>/local_naturalness/scores.json

scores.json schema:
  {
    "method":          "local_naturalness",
    "student_model":   "...",
    "local_m":         4,
    "n_problems":      <int>,
    "n_candidates":    <int>,
    "candidate_files": [{"fidx": 0, "teacher": "...", "path": "..."}, ...],
    "scores":          [[float, ...], ...]   # shape [C, N]
  }

Supports score cache (resume), sharding across GPUs, and merge mode.

Usage (single GPU):
  python score_local_naturalness_baseline.py \\
    --rsr_root      /home/tianruny/LIMO/data/training/rsr/RSR_data \\
    --out_root      /home/tianruny/LIMO/data/Q1 \\
    --student_model /home/tianruny/LIMO/models/students/qwen2.5-7b \\
    --students      qwen2.5-7b \\
    --batch_size    8 \\
    --local_m       4 \\
    --max_length    32768
"""

import os
import json
import glob
import argparse
import hashlib
from math import ceil
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_student_model(
    model_path: str,
    dtype: torch.dtype,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    print(f"[INFO] loading student model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    print(f"[INFO] model loaded (device_map=auto)")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Step splitting
# ---------------------------------------------------------------------------

def split_steps(asst_text: str) -> List[str]:
    steps = [s.strip() for s in asst_text.split("\n\n") if s.strip()]
    if len(steps) < 2:
        steps = [s.strip() for s in asst_text.split("\n") if s.strip()]
    if not steps:
        steps = [asst_text.strip()] if asst_text.strip() else [""]
    return steps


# ---------------------------------------------------------------------------
# Chat template helpers
# ---------------------------------------------------------------------------

def build_prompt_only(tokenizer, sys_c: str, user_c: str) -> str:
    messages = [
        {"role": "system", "content": sys_c},
        {"role": "user",   "content": user_c},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"{sys_c}\n\nUser: {user_c}\nAssistant:"


def build_context_text(
    tokenizer,
    sys_c: str,
    user_c: str,
    prev_steps: List[str],
) -> str:
    prompt_str = build_prompt_only(tokenizer, sys_c, user_c)
    if prev_steps:
        context_steps = "\n\n".join(prev_steps)
        return prompt_str + context_steps + "\n\n"
    return prompt_str


# ---------------------------------------------------------------------------
# Batched local naturalness scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_batch_local(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch_full_texts: List[str],
    batch_ctx_texts: List[str],
    max_length: int,
    device,
) -> List[float]:
    enc_ctx = tokenizer(
        batch_ctx_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    ctx_lengths = enc_ctx["attention_mask"].sum(dim=1).tolist()

    tokenizer.padding_side = "right"
    enc_full = tokenizer(
        batch_full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    input_ids      = enc_full["input_ids"].to(device)
    attention_mask = enc_full["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits  = outputs.logits                                              # [B, T, V]

    # Memory-efficient: F.cross_entropy does NOT materialise the full [B,T,V]
    # log-softmax tensor; it uses chunked CUDA kernels internally.
    targets         = input_ids[:, 1:]                                    # [B, T-1]
    logits_shifted  = logits[:, :-1, :].contiguous()                     # [B, T-1, V]
    B, T, V         = logits_shifted.shape
    shifted_lp      = -F.cross_entropy(                                   # [B, T-1]
        logits_shifted.view(B * T, V),
        targets.reshape(B * T),
        reduction="none",
    ).view(B, T)

    seq_lens = attention_mask.sum(dim=1)   # [B]

    scores: List[float] = []
    for b in range(len(batch_full_texts)):
        n_ctx   = int(ctx_lengths[b])
        seq_len = int(seq_lens[b].item())
        lp_start = n_ctx - 1
        lp_end   = seq_len - 1
        if lp_start < 0 or lp_start >= lp_end:
            scores.append(0.0)
            continue
        scores.append(float(shifted_lp[b, lp_start:lp_end].mean().item()))

    return scores


# ---------------------------------------------------------------------------
# Main scoring routine
# ---------------------------------------------------------------------------

def score_local_naturalness(
    rsr_root: str,
    out_path: str,
    student_model: str,
    n_problems: int = 5000,
    local_m: int = 4,
    max_length: int = 32768,
    batch_size: int = 8,
    verify_prompts: bool = True,
    score_cache_path: Optional[str] = None,
    shard_id: int = 0,
    num_shards: int = 1,
    merge_and_score: bool = False,
):
    method = "local_naturalness"
    cand_files = discover_candidate_files(rsr_root)
    C = len(cand_files)
    if C < 2:
        raise RuntimeError(f"Too few candidate files discovered: {C}")
    print(f"[INFO] discovered {C} candidate files")
    print(f"[INFO] local context window m = {local_m}")

    canonical_hashes: Optional[List[str]] = None
    if verify_prompts:
        canonical_hashes = build_canonical_prompt_hashes(cand_files[0][1], n_problems)
        print("[INFO] built canonical prompt hashes from:", cand_files[0][1])

    # ------------------------------------------------------------------
    # Score cache (resume on failure)
    # ------------------------------------------------------------------
    NAN = float("nan")
    score_cache: Dict[str, float] = {}
    if score_cache_path and os.path.isfile(score_cache_path):
        with open(score_cache_path, "r", encoding="utf-8") as f:
            score_cache = json.load(f)
        print(f"[INFO] loaded {len(score_cache)} cached scores from {score_cache_path}")

    scores: List[List[float]] = [[NAN] * n_problems for _ in range(C)]
    for key, val in score_cache.items():
        fi, pi = key.split(":")
        fi, pi = int(fi), int(pi)
        if 0 <= fi < C and 0 <= pi < n_problems:
            scores[fi][pi] = val

    # ------------------------------------------------------------------
    # MERGE mode
    # ------------------------------------------------------------------
    if merge_and_score:
        cache_dir = (
            os.path.dirname(score_cache_path)
            if score_cache_path
            else os.path.dirname(out_path)
        )
        for sid in range(num_shards):
            sp = os.path.join(cache_dir, f"score_cache_shard{sid}.json")
            if not os.path.isfile(sp):
                print(f"[WARN] shard cache not found: {sp}")
                continue
            with open(sp, "r", encoding="utf-8") as f:
                sc = json.load(f)
            for key, val in sc.items():
                fi, pi = key.split(":")
                fi, pi = int(fi), int(pi)
                if 0 <= fi < C and 0 <= pi < n_problems:
                    scores[fi][pi] = val
            print(f"[INFO] merged {len(sc)} scores from shard {sid}: {sp}")

    # ------------------------------------------------------------------
    # PASS 1 – score every (fidx, pid) assigned to this shard
    # ------------------------------------------------------------------
    if not merge_and_score:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model, tokenizer = load_student_model(student_model, dtype)
        device = next(model.parameters()).device

        for fidx, (teacher, fp) in enumerate(cand_files):
            if num_shards > 1 and fidx % num_shards != shard_id:
                continue

            if all(scores[fidx][pid] == scores[fidx][pid] for pid in range(n_problems)):
                print(f"[PASS1 SKIP] {fidx:02d} {teacher} – all scores cached")
                continue

            print(f"[PASS1 LOAD] {fidx:02d} {teacher} :: {os.path.basename(fp)}")
            data = load_json_list(fp)
            if len(data) < n_problems:
                raise ValueError(f"{fp} has {len(data)} items, expected >= {n_problems}")

            # Collect all step-pairs for this file
            pending: List[Tuple[str, str, int]] = []
            step_sum:   List[float] = [0.0] * n_problems
            step_count: List[int]   = [0]   * n_problems

            for pid in range(n_problems):
                if scores[fidx][pid] == scores[fidx][pid]:  # cached
                    continue
                ex = data[pid]
                sys_c, user_c, asst_c = get_sua(ex["messages"])

                if verify_prompts and canonical_hashes is not None:
                    h = sha1_text(sys_c + "\n" + user_c)
                    if h != canonical_hashes[pid]:
                        raise RuntimeError(
                            f"Prompt mismatch at pid={pid} for file={fp}\n"
                            f"Expected hash={canonical_hashes[pid]}, got hash={h}"
                        )

                steps = split_steps(asst_c)
                for j, step_text in enumerate(steps):
                    prev_steps = steps[max(0, j - local_m): j]
                    ctx_text   = build_context_text(tokenizer, sys_c, user_c, prev_steps)
                    full_text  = ctx_text + step_text
                    pending.append((full_text, ctx_text, pid))

            n_batches_total = ceil(len(pending) / batch_size) if pending else 0
            print(f"  → {len(pending)} step-pairs → {n_batches_total} batches of {batch_size}")

            for batch_idx, batch_start in enumerate(range(0, len(pending), batch_size)):
                batch      = pending[batch_start: batch_start + batch_size]
                full_texts = [x[0] for x in batch]
                ctx_texts  = [x[1] for x in batch]
                pids_b     = [x[2] for x in batch]

                try:
                    batch_scores = score_batch_local(
                        model, tokenizer, full_texts, ctx_texts, max_length, device
                    )
                except torch.cuda.OutOfMemoryError:
                    # Batch too large for this sequence length → fall back one-by-one
                    torch.cuda.empty_cache()
                    print(f"  [OOM] batch {batch_idx} – retrying {len(batch)} samples one-by-one")
                    batch_scores = []
                    for ft, ct in zip(full_texts, ctx_texts):
                        try:
                            s = score_batch_local(
                                model, tokenizer, [ft], [ct], max_length, device
                            )
                            batch_scores.extend(s)
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            print(f"  [OOM] single sample still OOM – scoring as 0.0")
                            batch_scores.append(0.0)

                for pid_b, sc in zip(pids_b, batch_scores):
                    step_sum[pid_b]   += sc
                    step_count[pid_b] += 1

                steps_done = batch_start + len(batch)
                if (batch_idx + 1) % 5 == 0 or steps_done == len(pending):
                    pct = steps_done / max(len(pending), 1) * 100
                    print(f"  → step-pairs {steps_done}/{len(pending)} ({pct:.1f}%)")

            # Compute trajectory means and update cache
            for pid in range(n_problems):
                if scores[fidx][pid] == scores[fidx][pid]:  # already cached
                    continue
                sc = step_sum[pid] / step_count[pid] if step_count[pid] > 0 else 0.0
                scores[fidx][pid] = sc
                score_cache[f"{fidx}:{pid}"] = sc

            if score_cache_path:
                d = os.path.dirname(score_cache_path)
                if d:
                    ensure_dir(d)
                with open(score_cache_path, "w", encoding="utf-8") as fw:
                    json.dump(score_cache, fw, ensure_ascii=False)

            del data
            print(f"  → finished {teacher}")

        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Write scores.json with full [C][N] matrix
    # ------------------------------------------------------------------
    ensure_dir(os.path.dirname(out_path))
    out = {
        "method":          method,
        "student_model":   student_model,
        "local_m":         local_m,
        "n_problems":      n_problems,
        "n_candidates":    C,
        "candidate_files": [
            {"fidx": i, "teacher": t, "path": p}
            for i, (t, p) in enumerate(cand_files)
        ],
        "scores": scores,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    n_nan = sum(1 for row in scores for v in row if v != v)
    print(f"[OK] wrote {C} × {n_problems} scores ({n_nan} NaN/unscored)")
    print(f"[OK] {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Q1 Local Naturalness baseline: step-level log-prob with local context."
    )
    ap.add_argument("--rsr_root",      type=str, required=True)
    ap.add_argument("--out_root",      type=str, required=True,
                    help="Q1 output root, e.g. /home/tianruny/LIMO/data/Q1")
    ap.add_argument("--student_model", type=str,
                    default="/home/tianruny/LIMO/models/students/qwen2.5-7b")
    ap.add_argument("--students",      type=str, required=True)
    ap.add_argument("--n_problems",    type=int, default=5000)
    ap.add_argument("--local_m",       type=int, default=4)
    ap.add_argument("--max_length",    type=int, default=32768)
    ap.add_argument("--batch_size",    type=int, default=8)
    ap.add_argument("--no_verify_prompts", action="store_true")
    ap.add_argument("--score_cache",   type=str, default=None)
    ap.add_argument("--num_shards",    type=int, default=1)
    ap.add_argument("--shard_id",      type=int, default=0)
    ap.add_argument("--merge_and_score", action="store_true",
                    help="Merge all shard caches and write scores.json.")
    args = ap.parse_args()

    students       = [s.strip() for s in args.students.split(",") if s.strip()]
    verify_prompts = not args.no_verify_prompts

    if args.shard_id >= args.num_shards:
        raise ValueError(f"--shard_id {args.shard_id} must be < --num_shards {args.num_shards}")

    for s in students:
        out_dir  = os.path.join(args.out_root, s, "local_naturalness")
        out_path = os.path.join(out_dir, "scores.json")

        if args.score_cache is not None:
            cache_path = args.score_cache
        elif args.num_shards > 1 and not args.merge_and_score:
            cache_path = os.path.join(out_dir, f"score_cache_shard{args.shard_id}.json")
        else:
            cache_path = os.path.join(out_dir, "score_cache.json")

        print(f"\n=== student: {s} ===")
        score_local_naturalness(
            rsr_root=args.rsr_root,
            out_path=out_path,
            student_model=args.student_model,
            n_problems=args.n_problems,
            local_m=args.local_m,
            max_length=args.max_length,
            batch_size=args.batch_size,
            verify_prompts=verify_prompts,
            score_cache_path=cache_path,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            merge_and_score=args.merge_and_score,
        )


if __name__ == "__main__":
    main()
