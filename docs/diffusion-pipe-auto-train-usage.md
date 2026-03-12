# Diffusion Pipe Auto Train Usage Guide

## Status

Version: `v1.0`

This is the first public release of the `diffusion-pipe-auto-train` skill. It is intentionally narrow and stable: it automates the highest-value pieces of `diffusion-pipe` image and image-edit training, but it does not yet cover the full dataset lifecycle.

## What This Skill Does

`diffusion-pipe-auto-train` provides a reproducible automation layer on top of [`diffusion-pipe`](https://github.com/tdrussell/diffusion-pipe):

- prepares a fixed workspace layout
- enforces stable image and edit dataset paths
- generates dataset TOML files
- generates training TOML files
- applies conservative VRAM-aware presets
- launches cache generation and DeepSpeed training

It currently supports these automatic paths:

- image LoRA: `flux-dev`, `qwen-image`, `sdxl`, `lumina2`, `hunyuanimage-2.1`
- image full finetune: `sdxl`, `lumina2`
- edit LoRA: `flux-kontext`, `qwen-image-edit`

## Runtime Requirement

Run this skill only in Linux or WSL2.

`diffusion-pipe` is built around DeepSpeed pipeline parallelism and is not a practical native Windows training stack. For that reason, the bundled `bootstrap`, `prepare`, and `launch` commands are all designed for Linux-style paths.

## Workspace Layout

Under `--workspace-root`, the script creates and reuses:

```text
data/image/train
data/image/eval
data/edit/train/target
data/edit/train/control
data/edit/eval/target
data/edit/eval/control
configs/datasets
configs/train
configs/manifests
runs
logs
```

For image training:

- place training images in `data/image/train`
- place optional eval images in `data/image/eval`
- keep captions as `.txt` files with matching stems

For edit training:

- place target images in `data/edit/.../target`
- place control or reference images in `data/edit/.../control`
- keep file stems matched one-to-one

## Standard Workflow

### 1. Bootstrap the training repo

```bash
python scripts/diffusion_pipe_auto.py bootstrap \
  --repo-root /workspace/diffusion-pipe \
  --install-latest-diffusers
```

Use `--install-latest-diffusers` for `qwen-image` and `qwen-image-edit`.

### 2. Prepare configs

Example: Flux Dev image LoRA on a 24GB GPU

```bash
python scripts/diffusion_pipe_auto.py prepare \
  --workspace-root /workspace/dp-auto \
  --model flux-dev \
  --train-kind image \
  --mode lora \
  --epochs 16 \
  --num-gpus 1 \
  --gpu-vram-gb 24 \
  --model-arg diffusers_path=/models/FLUX.1-dev
```

Example: Lumina 2 full finetune on 24GB

```bash
python scripts/diffusion_pipe_auto.py prepare \
  --workspace-root /workspace/dp-auto \
  --model lumina2 \
  --train-kind image \
  --mode full \
  --epochs 8 \
  --num-gpus 1 \
  --gpu-vram-gb 24 \
  --model-arg transformer_path=/models/lumina_2_model_bf16.safetensors \
  --model-arg llm_path=/models/gemma_2_2b_fp16.safetensors \
  --model-arg vae_path=/models/flux_vae.safetensors
```

### 3. Launch training

```bash
python scripts/diffusion_pipe_auto.py launch \
  --repo-root /workspace/diffusion-pipe \
  --manifest /workspace/dp-auto/configs/manifests/<run-name>.json \
  --cache-first
```

Optional resume examples:

```bash
python scripts/diffusion_pipe_auto.py launch \
  --repo-root /workspace/diffusion-pipe \
  --manifest /workspace/dp-auto/configs/manifests/<run-name>.json \
  --resume-latest
```

```bash
python scripts/diffusion_pipe_auto.py launch \
  --repo-root /workspace/diffusion-pipe \
  --manifest /workspace/dp-auto/configs/manifests/<run-name>.json \
  --resume-run 20250312_07-06-40
```

## What v1.0 Optimizes For

- stable defaults over open-ended tuning
- low operational overhead
- explicit dataset layout
- reproducible config generation
- safe VRAM floors for common image training paths

## Current Limitations

`v1.0` does not yet automate:

- dataset download
- dataset cleaning
- dataset deduplication
- caption quality checks
- automatic tagging
- automatic training set organization from raw source folders
- full raw-data-to-training orchestration

## Planned Roadmap After v1.0

The next optimization stage is intended to expand the skill from a training launcher into a fuller dataset-to-training pipeline:

1. Automatic dataset download from predefined or user-supplied sources.
2. Automated dataset cleaning, including corrupt-file checks, simple filtering, and duplicate handling.
3. Automated tagging or caption generation for image and edit datasets.
4. Automatic organization of training-ready dataset folders.
5. End-to-end automation that prepares, validates, and launches training from those organized datasets.

## References

- [Skill definition](../skills/diffusion-pipe-auto-train/SKILL.md)
- [Preset reference](../skills/diffusion-pipe-auto-train/references/presets.md)
- [Release paper](./diffusion-pipe-auto-train-paper.md)
