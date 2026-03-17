# LIMO: Math Reasoning SFT

Fine-tune **Qwen2.5-Math-7B** on provided training data and evaluate on AIME/AMC/MATH500.

## Model

Download **Qwen2.5-Math-7B** from Hugging Face:

```bash
huggingface-cli download Qwen/Qwen2.5-Math-7B --local-dir models/students/qwen2.5-math-7b
```

## Environment

```bash
conda create -n LIMO python=3.11
conda activate LIMO
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers peft accelerate tqdm numpy scipy matplotlib pyyaml
pip install deepspeed liger-kernel
pip install flash-attn --no-build-isolation
```

## Training

Place the provided `train.jsonl` anywhere, then run:

```bash
torchrun --nproc_per_node=4 train/train_full.py \
    --train_data  /path/to/train.jsonl \
    --model_path  models/students/qwen2.5-math-7b \
    --output_dir  results/checkpoints/run1
```

**Hyperparameters** (edit `train/configs/sft_full.yaml` to override):

| Parameter | Default |
|-----------|---------|
| learning_rate | 1e-5 |
| epochs | 10 |
| global batch size | 64 (4 GPU × 1 × 16) |
| max_seq_length | 32768 |
| precision | bf16 |
| attention | Flash Attention 2 |
| multi-GPU | ZeRO-3 (DeepSpeed) |

Training outputs:
- `results/checkpoints/run1/loss_curve.png` — live train + eval loss plot
- `results/checkpoints/run1/loss_log.csv`
- `results/checkpoints/run1/final/` — saved model

## Evaluation

```bash
# Step 1: Generate answers
python eval/generate.py \
    --model_path  results/checkpoints/run1/final \
    --benchmark_path data/benchmarks/AIME.json \
    --output_dir  results/eval/run1/AIME \
    --benchmark   AIME

# Step 2: Score
python eval/score.py \
    --output_dir results/eval/run1/AIME \
    --benchmark  AIME
```

Results saved to `results/eval/run1/AIME/summary.json`.
