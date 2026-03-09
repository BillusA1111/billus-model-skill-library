# HF, Accelerate, DeepSpeed, And FSDP

## Use This Reference For

- Hugging Face Trainer or custom Accelerate training loops
- DeepSpeed launchers, ds config files, ZeRO stages, FSDP wrapping, checkpoint save or resume behavior
- Scheduler, optimizer, batch-size, gradient-accumulation, and mixed-precision changes

## Audit Order

1. Find the real launch entrypoint.
2. Trace CLI args or config merge order.
3. Find where the model, optimizer, scheduler, and dataloaders are created.
4. Find how distributed wrapping happens.
5. Find how checkpoints and resumed scheduler state are loaded.

## Framework Notes

### Hugging Face Trainer

- Check whether values come from `TrainingArguments`, a custom dataclass, or a wrapper config layer.
- Verify whether callbacks, evaluation cadence, save cadence, and reporting integrations depend on the changed argument.
- If gradient checkpointing or flash attention are toggled, inspect model-specific guards.

### Accelerate custom loops

- Confirm whether batch size is per device or effective global batch size.
- Check where `accelerator.prepare()` is called and what objects are wrapped.
- Verify gradient accumulation is handled once, not once in code and once in config.

### DeepSpeed

- Trace ownership of optimizer and scheduler settings; they may live in Python config or the ds json.
- Check ZeRO stage, offload, and bucket settings before changing batch size or precision.
- Resume behavior can silently differ across ZeRO stages; inspect checkpoint load code rather than assuming parity.

### FSDP

- Audit auto-wrap policy, ignored modules, mixed precision policy, and state-dict type.
- Model changes that alter module boundaries can break wrap assumptions and checkpoint materialization.

## Common Failure Modes

- Changing batch size in Python while the launcher still overrides it
- Changing LR without matching warmup or total steps logic
- Assuming DeepSpeed json settings are active when code later overrides them
- Treating resumed training as comparable after changing scheduler or optimizer groups
- Breaking checkpoint save or load because wrapper policy changed with the model graph
