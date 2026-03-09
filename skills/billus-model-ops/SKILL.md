---
name: billus-model-ops
description: Framework-aware workflow for model-engineering tasks in common LLM, VL, and image-generation repositories. Use when Codex needs to inspect or modify training code, config files, launch scripts, model definitions, tokenizer or image-processor settings, checkpoint utilities, PEFT adapters, diffusion pipelines, evaluation harnesses, or experiment notes in PyTorch, Hugging Face Trainer, Accelerate, DeepSpeed, FSDP, PEFT, diffusers, LLaVA-like, Qwen-VL-like, or InternVL-like stacks.
---

# Billus Model Ops

## Overview

Use this skill for the model repos you touch every day: Hugging Face training stacks, DeepSpeed or FSDP scale-up flows, PEFT or QLoRA adaptation, VL wiring, and diffusion-style image generation or editing systems. Start from repository evidence, identify the active framework path, make the smallest viable change, and leave behind validation notes another engineer can trust.

## Quick Start

1. Identify the framework stack.
   - Run `python scripts/detect_training_stack.py <repo-root>` when the repo is unfamiliar.
   - Confirm the actual launch path with `rg -n "accelerate launch|deepspeed|torchrun|Trainer\\(|DiffusionPipeline|LoraConfig|vision_tower|projector|ControlNet" <repo-root>`.
2. Classify the task.
   - Trainer, scheduler, batch size, optimizer, data path: read `references/frameworks-hf-accelerate-deepspeed.md`
   - LoRA, QLoRA, adapter merge, low-bit finetune: read `references/frameworks-peft-and-lowbit.md`
   - LLaVA, Qwen-VL, InternVL, projector, image tokens, visual encoder: read `references/frameworks-vl-stacks.md`
   - diffusers, image edit, inpainting, ControlNet, IP-Adapter, scheduler, VAE: read `references/frameworks-diffusers-image.md`
   - Before final delivery, always read `references/validation-and-release.md`
3. Keep the change bounded.
   - Prefer modifying the existing config or launch surface instead of introducing duplicate knobs.
   - Keep baseline seeds, datasets, eval prompts, and sample grids stable unless the task explicitly changes them.
4. Validate before handoff.
   - Use the lightest repo-native smoke test that still exercises the changed path.
   - Summarize what was validated, what still needs a full run, and the main rollback lever.

## Workflow

### 1. Map the active stack

Identify:

- Trainer style: Hugging Face Trainer, raw PyTorch loop, Accelerate custom loop, DeepSpeed launcher, FSDP wrapper, diffusers trainer, or mixed custom code
- Config surface: argparse, dataclass, YAML, JSON, Hydra, shell scripts, environment variables
- Model family: decoder-only LLM, encoder-decoder, VL with vision tower plus projector, diffusion or flow image model
- Runtime topology: single GPU, DDP, ZeRO, FSDP, CPU offload, mixed precision, low-bit adapters

Do not infer the stack from dependencies alone. Trace from launch command to parser to model construction.

### 2. Convert the user request into a change class

Use one of these buckets:

- Hyperparameter or data-pipeline edit
- Scale-up or checkpoint migration
- PEFT or low-bit finetuning change
- VL connector or multimodal formatting change
- Image-generation or image-edit pipeline change
- Distillation, pruning, or quantization change
- Validation-only or experiment-analysis task

State the likely blast radius: config only, model code, checkpoint tooling, eval path, export path, or all of them.

### 3. Preserve reproducibility

- Keep a clear baseline versus candidate record.
- If a new flag is added, thread it through parser, config dump, logging, and docs/comments together.
- For visual models, keep prompt sets and image seeds fixed when comparing outputs.
- Use `scripts/new_experiment_note.py` for a compact experiment plan.
- Use `scripts/summarize_training_log.py` before making the next tuning move.

### 4. Validate at the correct depth

- Config edits: parse check, merged config dump, one-step or one-batch smoke run
- Scale-up or migration: instantiate model, exercise checkpoint load, run one forward pass
- PEFT or low-bit: verify target modules, adapter load/merge path, and dtype boundaries
- VL: verify image token count, projector path, masking, and one multimodal generation sample
- diffusers or image editing: verify scheduler path, prompt-to-image or image-edit call, and fixed-seed sample grid

Never claim a model change is safe because code "looks aligned." Exercise the changed path.

## References

- `references/frameworks-hf-accelerate-deepspeed.md`
  - Trainer, Accelerate, DeepSpeed, FSDP, optimizer, scheduler, and config-surface edits.
- `references/frameworks-peft-and-lowbit.md`
  - LoRA, QLoRA, PEFT adapter flows, merge rules, and low-bit finetuning cautions.
- `references/frameworks-vl-stacks.md`
  - LLaVA-like, Qwen-VL-like, InternVL-like multimodal stacks and connector audits.
- `references/frameworks-diffusers-image.md`
  - diffusers, image generation, instruction editing, inpainting, ControlNet, and VAE or scheduler changes.
- `references/validation-and-release.md`
  - Final acceptance checks, reproducibility notes, and rollback framing.

## Scripts

- `scripts/detect_training_stack.py`
  - Inspect a repo and summarize likely frameworks, launch surfaces, and model-family hints.
- `scripts/summarize_training_log.py`
  - Parse common training log patterns into markdown or JSON summaries.
- `scripts/new_experiment_note.py`
  - Generate a compact markdown experiment note for baseline, hypothesis, metrics, and commands.
