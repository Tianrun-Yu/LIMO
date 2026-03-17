# LIMO: Trajectory Selection for Math Reasoning SFT

This repo implements a pipeline for selecting high-quality teacher trajectories to fine-tune a student LLM on competition math (AIME/AMC).

## Pipeline Overview

```
RSR_data (teacher trajectories)
    │
    ▼
[Q1] Score all (candidate, problem) pairs
    │  · RSR baseline         (Q1/score_rsr_baseline.py)
    │  · topk_entropy         (Q1/score_topk_entropy_baseline.py)
    │  · global_naturalness   (Q1/score_global_naturalness_baseline.py)
    │
    ▼
[Q2] Select one trajectory per problem
    │  · chi2_B{B}  — χ²-optimal deterministic top-B  (Q2/select_chi2.py)
    │  · topb_B{B}  — uniform top-B                   (Q2/select_topb.py)
    │  · kl_tau{τ}  — KL-regularized                  (Q2/select_kl.py)
    │
    ▼
[Train] Full-parameter SFT
    │  train/train_full.py   (Flash Attn 2, ZeRO-3, Packing, Liger, Grad Ckpt)
    │  train/train_lora.py   (LoRA version)
    │
    ▼
[Eval] AIME / AMC / MATH500
       eval/generate.py → eval/extract_answer.py → eval/score.py
```

## Model

Student model: **Qwen2.5-Math-7B**
Download from Hugging Face: `Qwen/Qwen2.5-Math-7B`

```bash
huggingface-cli download Qwen/Qwen2.5-Math-7B --local-dir models/students/qwen2.5-math-7b
```

## Environment

```bash
conda create -n LIMO python=3.11
conda activate LIMO
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers peft accelerate datasets tqdm numpy scipy matplotlib pyyaml
pip install deepspeed liger-kernel
pip install flash-attn --no-build-isolation
```

## Quick Start

### Step 1 — Q1 Scoring (score all candidates)

```bash
# Two-GPU sharded run, then merge
CUDA_VISIBLE_DEVICES=0 python Q1/score_rsr_baseline.py \
    --rsr_root data/training/rsr/RSR_data \
    --out_root data/Q1 \
    --student_model models/students/qwen2.5-math-7b \
    --students qwen2.5-math-7b \
    --num_shards 2 --shard_id 0 &

CUDA_VISIBLE_DEVICES=1 python Q1/score_rsr_baseline.py \
    --rsr_root data/training/rsr/RSR_data \
    --out_root data/Q1 \
    --student_model models/students/qwen2.5-math-7b \
    --students qwen2.5-math-7b \
    --num_shards 2 --shard_id 1 &

wait

python Q1/score_rsr_baseline.py \
    --rsr_root data/training/rsr/RSR_data \
    --out_root data/Q1 \
    --student_model models/students/qwen2.5-math-7b \
    --students qwen2.5-math-7b \
    --num_shards 2 --merge_and_score
```

### Step 2 — Q2 Selection

```bash
# χ²-optimal top-5 selection (no τ hyperparameter)
python Q2/select_chi2.py \
    --q1_root  data/Q1 \
    --q2_root  data/Q2 \
    --rsr_root data/training/rsr/RSR_data \
    --students qwen2.5-math-7b \
    --q1_baseline rsr \
    --B 5
```

Output: `data/Q2/qwen2.5-math-7b/rsr/chi2_B5/train.jsonl`

### Step 3 — Full-Parameter SFT

```bash
torchrun --nproc_per_node=4 train/train_full.py \
    --train_data data/Q2/qwen2.5-math-7b/rsr/chi2_B5/train.jsonl \
    --model_path models/students/qwen2.5-math-7b \
    --output_dir results/checkpoints/qwen2.5-math-7b/rsr/chi2_B5_full
```

Training monitors AIME eval loss in real time:
- `results/.../loss_curve.png` — live loss plot
- `results/.../loss_log.csv`   — step, train_loss, eval_loss

### Step 4 — Evaluation

```bash
# Generate answers
python eval/generate.py \
    --model_path results/checkpoints/qwen2.5-math-7b/rsr/chi2_B5_full/final \
    --benchmark_path data/benchmarks/AIME.json \
    --output_dir results/eval/qwen2.5-math-7b/rsr/chi2_B5_full/AIME \
    --benchmark AIME

# Score
python eval/score.py \
    --output_dir results/eval/qwen2.5-math-7b/rsr/chi2_B5_full/AIME \
    --benchmark AIME
```

## Key Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| learning_rate | 1e-5 |
| epochs | 10 |
| global batch size | 64 |
| max_seq_length | 32768 |
| method | Full fine-tune |
| attention | Flash Attention 2 |
| precision | bf16 |
| multi-GPU | ZeRO-3 (DeepSpeed) |
| packing | greedy bin-packing + position_ids reset |
| kernels | Liger (fused RMSNorm, SwiGLU, CE) |

## Data Format

`train.jsonl` — one JSON object per line:
```json
{
  "messages": [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "problem text"},
    {"role": "assistant", "content": "step-by-step solution ... \\boxed{42}"}
  ],
  "meta": {
    "problem_id": 0,
    "teacher": "deepseek-r1-0528",
    "q2_method": "chi2_B5",
    "train_weight": 0.23
  }
}
```

## LESS Scoring (optional)

Computes cosine similarity between training gradients and AIME eval gradients:

```bash
CUDA_VISIBLE_DEVICES=0 python less/less_score.py \
    --model  models/students/qwen2.5-math-7b \
    --data   data/training/rsr/RSR_data \
    --eval   data/benchmarks/AIME.json \
    --output less/scores \
    --gpu_id 0 --n_gpus 2 &

CUDA_VISIBLE_DEVICES=1 python less/less_score.py \
    --model  models/students/qwen2.5-math-7b \
    --data   data/training/rsr/RSR_data \
    --eval   data/benchmarks/AIME.json \
    --output less/scores \
    --gpu_id 1 --n_gpus 2
```
