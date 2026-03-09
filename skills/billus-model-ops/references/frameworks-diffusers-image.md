# Diffusers And Image Pipelines

## Use This Reference For

- diffusers-based training or inference repos
- Text-to-image, image-to-image, inpainting, image editing, ControlNet, IP-Adapter, or LoRA changes
- Scheduler, UNet or transformer denoiser, VAE, and conditioning-path edits

## Audit Path

1. Pipeline assembly
2. Text and optional image conditioning
3. Backbone denoiser
4. Scheduler or sampler
5. VAE encode-decode path
6. Save or export path
7. Visual evaluation scripts

## Key Checks

### Scheduler edits

- Distinguish training noise schedule from inference scheduler selection.
- Match step count, guidance scale, and sigma parameterization when comparing quality.

### VAE and latent path

- Verify scaling factors, image normalization, and decode dtype.
- Image edit and inpainting flows are especially sensitive to latent encode-decode mismatches.

### ControlNet and adapters

- Trace control image preprocessing, resize rules, and conditioning strength.
- Confirm adapter or control modules are saved and loaded with the expected naming conventions.

### Image editing

- Keep fixed source images, masks, and seeds when comparing output changes.
- Include at least one locality-preservation test and one prompt-following test.

## Common Failure Modes

- Scheduler mismatch between training artifacts and inference pipeline
- LoRA target modules no longer matching the backbone after a library update
- Mask alignment bugs in inpainting or editing flows
- Calling a visual improvement based on different seeds instead of controlled comparisons
