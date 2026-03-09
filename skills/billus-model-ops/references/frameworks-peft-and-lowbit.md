# PEFT And Low-Bit Finetuning

## Use This Reference For

- LoRA, QLoRA, DoRA, adapter-only finetuning, merge or unmerge flows
- bitsandbytes low-bit loading, target-module selection, rank or alpha changes
- Requests to finetune larger models on smaller hardware

## Audit Surface

- Base model load path
- Quantization config
- PEFT config and target modules
- Trainable parameter filtering
- Save, load, merge, and export paths

## Key Checks

### Target modules

- Confirm the module names actually exist in the current model class.
- Distinguish attention projections, MLP projections, vision-projector modules, and diffusion backbone blocks.
- Refactors often break stale target-module lists silently.

### QLoRA and low-bit

- Separate storage dtype from compute dtype.
- Check whether norms, embeddings, or lm head remain higher precision by design.
- Verify whether gradient checkpointing and low-bit adapters interact cleanly in the current stack.

### Merge behavior

- Check whether downstream inference expects merged weights, adapter folders, or both.
- If multiple adapters are supported, confirm naming and loading conventions before changing save paths.

## Common Failure Modes

- Training zero parameters because the target-module list missed the renamed modules
- Exporting only base weights after finetuning adapters
- Merging adapters into a quantized base path that does not support the intended precision
- Comparing runs with different base checkpoints and calling it a pure adapter change
