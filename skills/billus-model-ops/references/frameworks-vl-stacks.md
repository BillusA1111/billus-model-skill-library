# VL Framework Stacks

## Use This Reference For

- LLaVA-like, Qwen-VL-like, InternVL-like, and similar vision-language repositories
- Vision tower swaps, projector changes, image-token handling, multimodal chat templates, or visual-data pipeline edits

## Audit Path

1. Image preprocessing and processor config
2. Vision encoder output shape and freeze policy
3. Projector, resampler, or connector
4. Tokenizer and multimodal special tokens
5. Label masking and conversation formatting
6. Generation and evaluation path

## Common Stack-Specific Checks

### LLaVA-like stacks

- Look for explicit image placeholder tokens and conversation templates.
- Verify projector output length and connector dtype assumptions.
- Check whether the vision tower is frozen and where that policy is enforced.

### Qwen-VL-like stacks

- Audit processor or tokenizer glue code carefully; image tags and message formatting often matter as much as model code.
- Check whether multi-image prompts or bounding-box style outputs depend on prompt serialization utilities.

### InternVL-like stacks

- Confirm patch sampling, dynamic resolution, and image tiling behavior.
- Check whether the repo carries separate image processor and chat formatting helpers that must be updated together.

## Common Failure Modes

- Special multimodal tokens changed in prompts but not tokenizer files
- Image-token count no longer matching connector output
- Eval drift caused by changed image resize or crop policy
- Visual backbone swap succeeding at load time but producing semantically mismatched features
