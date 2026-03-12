---
name: diffusion-pipe-auto-train
description: Automate diffusion-pipe image or image-edit training with stable LoRA presets and selected full finetune presets. Use when Codex needs to prepare a WSL or Linux diffusion-pipe workspace, enforce fixed dataset paths, generate TOML configs, bootstrap dependencies, or launch VRAM-aware training runs for Flux, Qwen-Image, SDXL, Lumina 2, or HunyuanImage.
---

# Diffusion Pipe Auto Train

## Overview

Use this skill when the user wants a mostly hands-off `diffusion-pipe` training flow for image or edit datasets. It bundles a WSL-first bootstrap and launcher script, a fixed workspace layout, and conservative presets optimized for stable training on commodity GPUs instead of open-ended hyperparameter hunting.

## Version

Current release: `v1.0`

This `v1.0` release focuses on stable automation for preparing configs, enforcing dataset layout, and launching image or image-edit training runs on top of `diffusion-pipe`.

Planned optimizations after `v1.0`:

- automatic dataset download
- dataset cleaning and filtering
- automated tagging or caption generation
- automatic dataset organization before training
- a more complete end-to-end pipeline from raw data to training launch

## Scope

Supported automatic paths:

- Image LoRA: `flux-dev`, `qwen-image`, `sdxl`, `lumina2`, `hunyuanimage-2.1`
- Image full finetune: `sdxl`, `lumina2`
- Edit LoRA: `flux-kontext`, `qwen-image-edit`

Out of scope for the automatic path:

- Video training
- Native Windows training
- Experimental full finetunes for models that the repo supports but does not document with a stable low-VRAM recipe

If the user explicitly asks for an unsupported automatic path, explain that `diffusion-pipe` may support it, but this skill keeps the bundled automation on the stable surface.

## Quick Start

1. Confirm the runtime.
   - Use Linux or WSL2 only.
   - Do not attempt native Windows training. `diffusion-pipe` is built around DeepSpeed pipeline parallelism and the upstream repo documents Windows as impractical.
2. Normalize the request into:
   - `model`
   - `train_kind` as `image` or `edit`
   - `mode` as `lora` or `full`
   - `epochs`
   - `num_gpus`
   - `gpu_vram_gb`
   - required base model paths
3. Read `references/presets.md` if you need the support matrix, fixed dataset layout, model path requirements, or VRAM floors.
4. Run every bundled command inside Linux or WSL2, including `prepare`.
5. Bootstrap or update the training repo:
   ```bash
   python scripts/diffusion_pipe_auto.py bootstrap --repo-root /workspace/diffusion-pipe
   ```
6. Prepare configs and stable workspace directories:
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
7. Launch caching and training:
   ```bash
   python scripts/diffusion_pipe_auto.py launch \
     --repo-root /workspace/diffusion-pipe \
     --manifest /workspace/dp-auto/configs/manifests/flux-dev-image-lora.json \
     --cache-first
   ```
8. Validate:
   - confirm the generated dataset and train TOML files exist
   - confirm the launch command exits successfully
   - confirm a run directory is created under `<workspace-root>/runs`

## Fixed Workspace Layout

The script creates and reuses these stable paths:

- `<workspace-root>/data/image/train`
- `<workspace-root>/data/image/eval`
- `<workspace-root>/data/edit/train/target`
- `<workspace-root>/data/edit/train/control`
- `<workspace-root>/data/edit/eval/target`
- `<workspace-root>/data/edit/eval/control`
- `<workspace-root>/configs/datasets`
- `<workspace-root>/configs/train`
- `<workspace-root>/configs/manifests`
- `<workspace-root>/runs`
- `<workspace-root>/logs`

For `image` training, media files and matching `.txt` captions go under `data/image/train` and optionally `data/image/eval`.

For `edit` training:

- target images go under `data/edit/.../target`
- control or reference images go under `data/edit/.../control`
- target and control file stems must match one-to-one
- keep target and control aspect ratios similar, because `diffusion-pipe` resizes control inputs to the target bucket

## Workflow Rules

- Prefer LoRA when the user does not insist on full finetuning or when VRAM is tight.
- Enforce the VRAM floors in `references/presets.md`. If the request is below the floor, stop and say why.
- Keep the automation conservative:
  - `micro_batch_size_per_gpu = 1`
  - activation checkpointing on
  - regular eval and checkpoint cadence
  - low-VRAM presets use block swapping only where the upstream repo documents or demonstrates it
- For `qwen-image` and `qwen-image-edit`, bootstrap with the latest Diffusers from GitHub because the upstream docs call that out.
- For `lumina2` full finetune on 24GB, keep the documented `AdamW8bitKahan + gradient_release` recipe.
- If the user asks for a different resolution, LoRA rank, or optimizer, it is fine to override the preset, but call out that the bundled preset is no longer the default stable path.

## Scripts

- `scripts/diffusion_pipe_auto.py`
  - `bootstrap`: clone or update `diffusion-pipe`, update submodules, install repo requirements, optionally install Flash Attention and latest Diffusers
  - `prepare`: create fixed dataset folders, write dataset and training TOML files, and write a JSON manifest with the recommended launch command
  - `launch`: optionally run cache-only first, then launch DeepSpeed training with the correct environment variables

## References

- `references/presets.md`
  - Support matrix
  - Fixed dataset layout
  - Required `--model-arg` keys by model
  - Conservative VRAM floors
  - Stable optimizer and resolution defaults
