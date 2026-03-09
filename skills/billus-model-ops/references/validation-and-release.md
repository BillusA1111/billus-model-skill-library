# Validation And Release

## Use This Reference For

- Final pass before handing back any framework-specific model change
- Preparing a change summary another engineer can run or roll back safely

## Minimum Acceptance

- The changed config path parses and resolves to the intended values
- The changed model path instantiates
- The changed runtime path executes at least one representative step
- The changed save or load path still works if in scope
- Baseline versus candidate comparison is controlled enough to mean something

## Visual-Model Extras

- For VL, run a tiny fixed prompt suite with representative images.
- For image generation or editing, keep prompts, source images, masks, and seeds fixed.
- Prefer before/after grids or explicit pass/fail notes over vague qualitative claims.

## Report Back In This Shape

- What changed
- Why it should help
- What was validated locally
- What still needs a full run
- Main regression risks
- Fast rollback lever
