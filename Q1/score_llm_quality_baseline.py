#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1 – LLM-judged Quality scorer.

Uses Qwen3-32B-Instruct (non-thinking mode) as a judge to score every
(candidate, problem) pair and writes the full C×N score matrix to scores.json.
Uses PyTorch built-in SDPA attention (no flash_attn install required).

Energy function:
  score(fidx, pid) = judge_overall_score ∈ [0.0, 1.0]
  (higher = better reasoning quality)

Output:
  out_root/<student>/llm_quality/scores.json

scores.json schema:
  {
    "method":          "llm_quality",
    "judge_model":     "...",
    "n_problems":      <int>,
    "n_candidates":    <int>,
    "candidate_files": [{"fidx": 0, "teacher": "...", "path": "..."}, ...],
    "scores":          [[float, ...], ...]   # shape [C, N]
  }

Supports score cache (resume), sharding across GPUs, and merge mode.

Usage (single GPU):
  python score_llm_quality_baseline.py \\
    --rsr_root      /home/tianruny/LIMO/data/training/rsr/RSR_data \\
    --out_root      /home/tianruny/LIMO/data/Q1 \\
    --judge_model   /home/tianruny/LIMO/models/judges/qwen3-32b \\
    --students      qwen2.5-7b \\
    --batch_size    32

Usage (sharded):
  CUDA_VISIBLE_DEVICES=N python score_llm_quality_baseline.py ... \\
    --num_shards 8 --shard_id N

Merge all shards and write scores.json:
  python score_llm_quality_baseline.py ... \\
    --num_shards 8 --merge_and_score
"""

import os
import json
import glob
import argparse
import hashlib
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = (
    "You are a meticulous and highly critical evaluator of AI reasoning. "
    "Your primary goal is to identify and quantify subtle flaws, logical gaps, "
    "inefficiencies, and hidden assumptions. Do not default to a high score. "
    "Your starting assumption should be critical, and you must rigorously justify "
    "every point awarded."
)

JUDGE_USER_TEMPLATE = """\
First, please carefully read the following problem statement: <Problem> {question} </Problem>

Now, please carefully read the following candidate's chain-of-thought reasoning: <Reasoning>
{reasoning_to_evaluate} </Reasoning>

When evaluating this reasoning, you must adhere to the following five key evaluation criteria and the scoring rubric below.

Scoring Guidelines and Calibration: You must use the full 0.0 to 1.0 scale. Scores should not be clustered at the top. Use this rubric to anchor your scores:
1.0 (Exceptional/Flawless): Reserved for reasoning that is not only correct but also elegant, insightful, and comprehensive. It is perfectly structured and leaves no room for doubt. This score should be exceedingly rare.
0.8 - 0.9 (Excellent but Imperfect): The core reasoning is valid and well-supported, but there may be very minor, superficial issues (e.g., a trivial typo in a formula that doesn't affect the outcome, a slightly awkward phrasing). The conclusion is unaffected.
0.5 - 0.7 (Competent but Flawed): The reasoning is generally on the right track but contains noticeable and nontrivial flaws.
0.2 - 0.4 (Poor): The reasoning contains fundamental flaws that largely invalidate the process or conclusion.
0.0 - 0.1 (Unacceptable): The reasoning is completely incorrect, irrelevant, nonsensical, or makes no meaningful attempt to solve the problem.

Evaluation Criteria: Factual Accuracy, Logical Rigor, Solution Completeness, Reasoning Efficiency, Presentation Quality.

For each criterion, give a score from 0.0 to 1.0 (in 0.1 increments) and a brief justification in JSON.
Your output must be a single, valid JSON object with:
{{
  "dimensional_evaluation": {{...}},
  "overall_score": <float between 0.0 and 1.0>,
  "overall_reason": "<concise summary>"
}}"""


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
# Prompt formatting and score parsing
# ---------------------------------------------------------------------------

def format_judge_prompt(tokenizer, question: str, reasoning: str) -> str:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": JUDGE_USER_TEMPLATE.format(
                question=question,
                reasoning_to_evaluate=reasoning,
            ),
        },
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,   # Qwen3 non-thinking mode
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def parse_judge_score(raw: str, fallback: float = 0.0) -> float:
    try:
        start = raw.find("{")
        end   = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return fallback
        return max(0.0, min(1.0, float(json.loads(raw[start: end + 1])["overall_score"])))
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Judge model loading and batched generation
# ---------------------------------------------------------------------------

def load_judge_model(judge_model: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[INFO] loading judge tokenizer from {judge_model}")
    tokenizer = AutoTokenizer.from_pretrained(judge_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[INFO] loading judge model (dtype={dtype}, attn=sdpa)")
    model = AutoModelForCausalLM.from_pretrained(
        judge_model,
        dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    print("[INFO] judge model loaded")
    return model, tokenizer


@torch.no_grad()
def judge_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_texts: List[str],
    max_new_tokens: int,
    max_input_len: int,
) -> List[str]:
    tokenizer.padding_side = "left"
    enc = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_len,
    )
    input_ids      = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    out_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_ids = out_ids[:, input_ids.shape[1]:]
    return tokenizer.batch_decode(new_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main scoring routine
# ---------------------------------------------------------------------------

def score_llm_quality(
    rsr_root: str,
    out_path: str,
    judge_model: str,
    n_problems: int = 5000,
    batch_size: int = 32,
    max_new_tokens: int = 512,
    max_input_len: int = 3584,
    verify_prompts: bool = True,
    score_cache_path: Optional[str] = None,
    shard_id: int = 0,
    num_shards: int = 1,
    merge_and_score: bool = False,
):
    method = "llm_quality"
    cand_files = discover_candidate_files(rsr_root)
    C = len(cand_files)
    if C < 2:
        raise RuntimeError(f"Too few candidate files discovered: {C}")
    print(f"[INFO] discovered {C} candidate files")

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
    # PASS 1 – format prompts and collect pending (fidx, pid, prompt)
    # ------------------------------------------------------------------
    pending: List[Tuple[int, int, str]] = []

    if not merge_and_score:
        print(f"[INFO] loading judge tokenizer from {judge_model}")
        tokenizer = AutoTokenizer.from_pretrained(judge_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

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

            for pid in range(n_problems):
                if scores[fidx][pid] == scores[fidx][pid]:   # cached
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

                prompt_str = format_judge_prompt(tokenizer, question=user_c, reasoning=asst_c)
                pending.append((fidx, pid, prompt_str))

            del data

    total     = len(pending)
    n_batches = (total + batch_size - 1) // batch_size
    print(f"[INFO] {total} pairs to judge → {n_batches} batches of {batch_size}")

    # ------------------------------------------------------------------
    # PASS 1.5 – load judge model and run inference
    # ------------------------------------------------------------------
    if pending:
        model, tokenizer = load_judge_model(judge_model)

        for b_idx in range(n_batches):
            batch   = pending[b_idx * batch_size: (b_idx + 1) * batch_size]
            prompts = [item[2] for item in batch]
            pct     = (b_idx + 1) / n_batches * 100
            print(f"[JUDGE] batch {b_idx + 1}/{n_batches} ({pct:.1f}%)  size={len(prompts)}")

            try:
                raw_outputs = judge_batch(model, tokenizer, prompts, max_new_tokens, max_input_len)
            except torch.cuda.OutOfMemoryError:
                print("[WARN] OOM on batch, retrying one-by-one")
                raw_outputs = []
                for p in prompts:
                    torch.cuda.empty_cache()
                    try:
                        out = judge_batch(model, tokenizer, [p], max_new_tokens, max_input_len)
                        raw_outputs.extend(out)
                    except torch.cuda.OutOfMemoryError:
                        print("[WARN] single item OOM, score=0.0")
                        raw_outputs.append("")

            for (fidx, pid, _), raw in zip(batch, raw_outputs):
                sc = parse_judge_score(raw, fallback=0.0)
                scores[fidx][pid] = sc
                score_cache[f"{fidx}:{pid}"] = sc

            if score_cache_path:
                ensure_dir(os.path.dirname(score_cache_path))
                with open(score_cache_path, "w", encoding="utf-8") as f:
                    json.dump(score_cache, f, ensure_ascii=False)

        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Write scores.json with full [C][N] matrix
    # ------------------------------------------------------------------
    ensure_dir(os.path.dirname(out_path))
    out = {
        "method":          method,
        "judge_model":     judge_model,
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
        description="Q1 LLM Quality baseline: LLM judge score per (candidate, problem)."
    )
    ap.add_argument("--rsr_root",       type=str, required=True)
    ap.add_argument("--out_root",       type=str, required=True,
                    help="Q1 output root, e.g. /home/tianruny/LIMO/data/Q1")
    ap.add_argument("--judge_model",    type=str,
                    default="/home/tianruny/LIMO/models/judges/qwen3-32b")
    ap.add_argument("--students",       type=str, required=True)
    ap.add_argument("--n_problems",     type=int, default=5000)
    ap.add_argument("--batch_size",     type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--max_input_len",  type=int, default=3584)
    ap.add_argument("--no_verify_prompts", action="store_true")
    ap.add_argument("--score_cache",    type=str, default=None)
    ap.add_argument("--num_shards",     type=int, default=1)
    ap.add_argument("--shard_id",       type=int, default=0)
    ap.add_argument("--merge_and_score", action="store_true",
                    help="Merge all shard caches and write scores.json.")
    args = ap.parse_args()

    students       = [s.strip() for s in args.students.split(",") if s.strip()]
    verify_prompts = not args.no_verify_prompts

    if args.shard_id >= args.num_shards:
        raise ValueError(f"--shard_id {args.shard_id} must be < --num_shards {args.num_shards}")

    for s in students:
        out_dir  = os.path.join(args.out_root, s, "llm_quality")
        out_path = os.path.join(out_dir, "scores.json")

        if args.score_cache is not None:
            cache_path = args.score_cache
        elif args.num_shards > 1 and not args.merge_and_score:
            cache_path = os.path.join(out_dir, f"score_cache_shard{args.shard_id}.json")
        else:
            cache_path = os.path.join(out_dir, "score_cache.json")

        print(f"\n=== student: {s} ===")
        score_llm_quality(
            rsr_root=args.rsr_root,
            out_path=out_path,
            judge_model=args.judge_model,
            n_problems=args.n_problems,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            max_input_len=args.max_input_len,
            verify_prompts=verify_prompts,
            score_cache_path=cache_path,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            merge_and_score=args.merge_and_score,
        )


if __name__ == "__main__":
    main()
