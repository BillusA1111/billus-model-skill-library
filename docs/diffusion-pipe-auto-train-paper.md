# Diffusion Pipe Auto Train v1.0

## Abstract

`diffusion-pipe-auto-train` is a skill-layer automation package built to make `diffusion-pipe` easier to use in repeatable image and image-edit training workflows. Version `1.0` focuses on one specific problem: converting a loosely specified training request into a stable, runnable `diffusion-pipe` workspace with fixed dataset paths, generated TOML configs, and VRAM-aware launch defaults. This release deliberately narrows scope to improve reliability. It does not yet automate dataset acquisition, cleaning, tagging, or full raw-data preparation.

## 1. Background

`diffusion-pipe` is powerful, but its training workflow assumes that the operator can correctly manage:

- model-specific path requirements
- dataset TOML files
- training TOML files
- DeepSpeed launch commands
- low-VRAM settings
- cache generation and resume flows

That flexibility is valuable, but it also creates repeated setup work and a large error surface for routine training tasks.

## 2. Problem Statement

In practical image-model operations, engineers often want a narrower path:

- choose a model family
- choose LoRA or full finetune
- set epoch count
- point to a base model
- place datasets in the expected folders
- launch training with defaults that are known to be stable

Without automation, these steps are easy to misconfigure. Common failure modes include wrong model-path combinations, inconsistent dataset layouts, under-specified edit datasets, and low-VRAM launches that fail before meaningful progress.

## 3. Design Goals

Version `1.0` was designed with these goals:

1. Keep the supported surface small and stable.
2. Prefer deterministic config generation over manual editing.
3. Enforce a fixed dataset layout.
4. Encode conservative VRAM-aware presets.
5. Separate bootstrap, prepare, and launch into explicit commands.
6. Preserve compatibility with upstream `diffusion-pipe` configuration patterns.

## 4. System Design

The skill consists of three main parts:

### 4.1 Skill instructions

The skill definition explains when to use the automation, which model families are supported, and which requests remain out of scope.

### 4.2 Preset reference

The preset reference defines:

- supported automatic modes
- required per-model path arguments
- conservative VRAM floors
- stable optimizer and resolution defaults
- fixed dataset directory conventions

### 4.3 Automation script

The bundled `diffusion_pipe_auto.py` script exposes:

- `bootstrap`
- `prepare`
- `launch`

This split keeps operations explicit and debuggable while still making the overall workflow close to one-command-per-stage.

## 5. Supported Scope in v1.0

Automatic support in `v1.0` includes:

- image LoRA: `flux-dev`, `qwen-image`, `sdxl`, `lumina2`, `hunyuanimage-2.1`
- image full finetune: `sdxl`, `lumina2`
- edit LoRA: `flux-kontext`, `qwen-image-edit`

Out of scope in `v1.0`:

- video automation
- native Windows training
- wide-open hyperparameter search
- automated data acquisition and preprocessing

## 6. Why This Is Marked v1.0

This release is intentionally labeled `v1.0` rather than presented as a finished end state.

It already solves a real operational problem: repeated setup for `diffusion-pipe` image training. But it is still the first stable packaging of that workflow. The current version optimizes for safe structure and repeatability, not for full dataset lifecycle automation.

## 7. Limitations

The current release does not yet automate:

- dataset download
- dataset cleaning
- dataset deduplication
- tagging or caption generation
- training-set restructuring from raw scraped or mixed folders
- quality scoring before launch

Those missing pieces are important for a truly end-to-end training pipeline, and they are the next major target for iteration.

## 8. Planned Optimization Roadmap

Post-`v1.0`, the intended improvements are:

1. Automatic dataset download.
2. Automated dataset cleaning and filtering.
3. Automated tagging or caption generation.
4. Automatic organization of the final training dataset.
5. Fully automated training orchestration from prepared dataset to launch.

This means the long-term direction is not only "generate configs and run training," but "take raw or semi-raw training assets and turn them into a training-ready pipeline with minimal manual intervention."

## 9. Conclusion

`diffusion-pipe-auto-train v1.0` is a pragmatic release: it narrows the `diffusion-pipe` training surface into a reproducible automation path for image and image-edit tasks. It should be understood as a stable first release, not a final platform. The next phase of work is centered on automating dataset download, cleaning, tagging, organization, and then tying those steps into the existing training launcher for a fuller end-to-end workflow.
