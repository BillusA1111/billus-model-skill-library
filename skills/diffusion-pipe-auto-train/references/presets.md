# Diffusion-Pipe Automation Presets

## Scope

This skill keeps the automatic training path narrow on purpose:

- image LoRA: `flux-dev`, `qwen-image`, `sdxl`, `lumina2`, `hunyuanimage-2.1`
- image full finetune: `sdxl`, `lumina2`
- edit LoRA: `flux-kontext`, `qwen-image-edit`

The upstream `diffusion-pipe` repo supports more models and more full finetune combinations, but this automation only includes the modes with a reasonable stability story for a reusable skill.

## Fixed Workspace Layout

The script uses these fixed dataset paths under `--workspace-root`:

- `data/image/train`
- `data/image/eval`
- `data/edit/train/target`
- `data/edit/train/control`
- `data/edit/eval/target`
- `data/edit/eval/control`

Generated files go here:

- `configs/datasets/<run-name>_train_dataset.toml`
- `configs/datasets/<run-name>_eval_dataset.toml`
- `configs/train/<run-name>.toml`
- `configs/manifests/<run-name>.json`
- `runs/<run-name>/...`

## Required Model Args

Pass model-specific paths as repeated `--model-arg key=value`.

### flux-dev

- required: `diffusers_path`
- optional: `transformer_path`

### flux-kontext

- required: `diffusers_path`
- optional: `transformer_path`

### qwen-image

- required: `diffusers_path`

### qwen-image-edit

- required: `diffusers_path`
- optional: `transformer_path`

### sdxl

- required: `checkpoint_path`

### lumina2

- required: `transformer_path`, `llm_path`, `vae_path`

### hunyuanimage-2.1

- required: `transformer_path`, `vae_path`, `text_encoder_path`, `byt5_path`

## Conservative VRAM Floors

These are the automation guardrails, not universal truths. They mix upstream repo documentation with conservative defaults where the repo does not provide an exact floor.

### Source-grounded floors from diffusion-pipe docs

- `sdxl` full finetune: `48GB`, or `2x24GB` with `pipeline_stages=2`
- `lumina2` full finetune: `24GB`
- `qwen-image` LoRA: `24GB` with block swapping and `expandable_segments`

### Conservative automation floors

- `flux-dev` image LoRA: `24GB`
- `flux-kontext` edit LoRA: `24GB`
- `qwen-image-edit` edit LoRA: `48GB`
- `sdxl` image LoRA: `24GB`
- `lumina2` image LoRA: `24GB`
- `hunyuanimage-2.1` image LoRA: `24GB`

When a floor is conservative rather than explicitly documented upstream, mention that in the handoff.

## Stable Default Presets

### Common defaults

- `micro_batch_size_per_gpu = 1`
- `gradient_clipping = 1.0`
- `warmup_steps = 100`
- `eval_every_n_epochs = 1`
- `eval_before_first_step = true`
- `eval_micro_batch_size_per_gpu = 1`
- `eval_gradient_accumulation_steps = 1`
- `checkpoint_every_n_minutes = 120`
- `activation_checkpointing = true`
- `partition_method = "parameters"`
- `save_dtype = "bfloat16"`

### Image dataset defaults

- aspect ratio bucketing enabled
- `min_ar = 0.5`
- `max_ar = 2.0`
- `num_ar_buckets = 9`
- `num_repeats = 1`

### Edit dataset defaults

- same aspect ratio bucket settings as image
- each training example must exist in both `target` and `control`
- prefer `512` resolution buckets unless the user explicitly wants a heavier run

## Model Presets

### flux-dev image LoRA

- resolution: `512`
- optimizer: `adamw_optimi`
- learning rate: `2e-5`
- low-VRAM extras: `blocks_to_swap = 8`, `gradient_accumulation_steps = 4`

### flux-kontext edit LoRA

- resolution: `512`
- optimizer: `adamw_optimi`
- learning rate: `2e-5`
- low-VRAM extras: `blocks_to_swap = 12`, `gradient_accumulation_steps = 4`

### qwen-image image LoRA

- resolution: `640`
- optimizer: `automagic`
- low-VRAM extras: `blocks_to_swap = 8`, `gradient_accumulation_steps = 4`
- launcher env: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### qwen-image-edit edit LoRA

- resolution: `512`
- optimizer: `automagic`
- `gradient_accumulation_steps = 4`
- launcher env: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### sdxl image LoRA

- resolution: `1024`
- optimizer: `adamw_optimi`
- optimizer learning rate: `2e-5`
- model learning rates: `unet_lr = 4e-5`, `text_encoder_1_lr = 2e-5`, `text_encoder_2_lr = 2e-5`

### sdxl image full finetune

- resolution: `1024`
- optimizer: `adamw_optimi`
- automation learning rate: `1e-5`
- automation model learning rates: `unet_lr = 1e-5`, `text_encoder_1_lr = 5e-6`, `text_encoder_2_lr = 5e-6`
- if the user has `2x24GB`, use `pipeline_stages = 2`

The SDXL full-finetune learning rates above are conservative automation defaults. The upstream docs document the VRAM requirement, but do not publish a single best FFT LR recipe.

### lumina2 image LoRA

- resolutions: `512`, `1024`
- caption prefix:
  `You are an assistant designed to generate high-quality images based on user prompts. <Prompt Start> `
- optimizer: `adamw_optimi`
- learning rate: `2e-5`

### lumina2 image full finetune

- resolutions: `512`, `1024`
- same caption prefix as LoRA
- optimizer: `AdamW8bitKahan`
- learning rate: `5e-6`
- `gradient_release = true`

These settings come directly from the upstream Lumina 2 notes in `docs/supported_models.md`.

### hunyuanimage-2.1 image LoRA

- resolution: `512`
- optimizer: `adamw_optimi`
- learning rate: `2e-5`

The upstream docs note that `1024` on HunyuanImage has similar compute to `512` on Flux or Qwen. The automation default stays at `512` for a safer floor, and should be raised only when the user wants more detail and has the VRAM budget.
