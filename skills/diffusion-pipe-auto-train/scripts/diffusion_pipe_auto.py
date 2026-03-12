#!/usr/bin/env python3
"""Automation helpers for diffusion-pipe image and image-edit training."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

COMMON_TRAIN_DEFAULTS = {
    "micro_batch_size_per_gpu": 1,
    "gradient_clipping": 1.0,
    "warmup_steps": 100,
    "eval_every_n_epochs": 1,
    "eval_before_first_step": True,
    "eval_micro_batch_size_per_gpu": 1,
    "eval_gradient_accumulation_steps": 1,
    "checkpoint_every_n_minutes": 120,
    "activation_checkpointing": True,
    "partition_method": "parameters",
    "save_dtype": "bfloat16",
    "steps_per_print": 1,
}

OPTIM_ADAMW_OPTIMI = {
    "type": "adamw_optimi",
    "lr": 2e-5,
    "betas": [0.9, 0.99],
    "weight_decay": 0.01,
    "eps": 1e-8,
}

OPTIM_LUMINA_FULL = {
    "type": "AdamW8bitKahan",
    "lr": 5e-6,
    "betas": [0.9, 0.99],
    "weight_decay": 0.01,
    "eps": 1e-8,
    "gradient_release": True,
}

MODEL_SPECS = {
    "flux-dev": {
        "train_kind": "image",
        "modes": {"lora"},
        "required_model_args": ["diffusers_path"],
        "optional_model_args": ["transformer_path"],
        "min_vram_gb": {"lora": 24},
        "dataset": {"resolutions": [512]},
        "train": {
            "caching_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "low_vram_gradient_accumulation_steps": 4,
            "low_vram_blocks_to_swap": 8,
        },
        "launch_env": {
            "NCCL_P2P_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
        },
        "model": {
            "type": "flux",
            "dtype": "bfloat16",
            "transformer_dtype": "float8",
            "flux_shift": True,
        },
        "lora_optimizer": OPTIM_ADAMW_OPTIMI,
    },
    "flux-kontext": {
        "train_kind": "edit",
        "modes": {"lora"},
        "required_model_args": ["diffusers_path"],
        "optional_model_args": ["transformer_path"],
        "min_vram_gb": {"lora": 24},
        "dataset": {"resolutions": [512]},
        "train": {
            "caching_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "low_vram_gradient_accumulation_steps": 4,
            "low_vram_blocks_to_swap": 12,
        },
        "launch_env": {
            "NCCL_P2P_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
        },
        "model": {
            "type": "flux",
            "dtype": "bfloat16",
            "transformer_dtype": "float8",
        },
        "lora_optimizer": OPTIM_ADAMW_OPTIMI,
    },
    "qwen-image": {
        "train_kind": "image",
        "modes": {"lora"},
        "required_model_args": ["diffusers_path"],
        "optional_model_args": [],
        "min_vram_gb": {"lora": 24},
        "dataset": {"resolutions": [640]},
        "train": {
            "caching_batch_size": 8,
            "gradient_accumulation_steps": 2,
            "low_vram_gradient_accumulation_steps": 4,
            "low_vram_blocks_to_swap": 8,
        },
        "launch_env": {
            "NCCL_P2P_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        "model": {
            "type": "qwen_image",
            "dtype": "bfloat16",
            "transformer_dtype": "float8",
            "timestep_sample_method": "logit_normal",
        },
        "lora_optimizer": {
            "type": "automagic",
            "weight_decay": 0.01,
        },
    },
    "qwen-image-edit": {
        "train_kind": "edit",
        "modes": {"lora"},
        "required_model_args": ["diffusers_path"],
        "optional_model_args": ["transformer_path"],
        "min_vram_gb": {"lora": 48},
        "dataset": {"resolutions": [512]},
        "train": {
            "caching_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "low_vram_gradient_accumulation_steps": 4,
        },
        "launch_env": {
            "NCCL_P2P_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        "model": {
            "type": "qwen_image",
            "dtype": "bfloat16",
            "transformer_dtype": "float8",
            "timestep_sample_method": "logit_normal",
        },
        "lora_optimizer": {
            "type": "automagic",
            "weight_decay": 0.01,
        },
    },
    "sdxl": {
        "train_kind": "image",
        "modes": {"lora", "full"},
        "required_model_args": ["checkpoint_path"],
        "optional_model_args": [],
        "min_vram_gb": {"lora": 24, "full": 48},
        "dataset": {"resolutions": [1024]},
        "train": {
            "caching_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "low_vram_gradient_accumulation_steps": 4,
        },
        "launch_env": {
            "NCCL_P2P_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
        },
        "model": {
            "type": "sdxl",
            "dtype": "bfloat16",
        },
        "lora_optimizer": OPTIM_ADAMW_OPTIMI,
        "full_optimizer": {
            "type": "adamw_optimi",
            "lr": 1e-5,
            "betas": [0.9, 0.99],
            "weight_decay": 0.01,
            "eps": 1e-8,
        },
        "lora_model_overrides": {
            "unet_lr": 4e-5,
            "text_encoder_1_lr": 2e-5,
            "text_encoder_2_lr": 2e-5,
        },
        "full_model_overrides": {
            "unet_lr": 1e-5,
            "text_encoder_1_lr": 5e-6,
            "text_encoder_2_lr": 5e-6,
        },
    },
    "lumina2": {
        "train_kind": "image",
        "modes": {"lora", "full"},
        "required_model_args": ["transformer_path", "llm_path", "vae_path"],
        "optional_model_args": [],
        "min_vram_gb": {"lora": 24, "full": 24},
        "dataset": {
            "resolutions": [512, 1024],
            "caption_prefix": (
                "You are an assistant designed to generate high-quality images "
                "based on user prompts. <Prompt Start> "
            ),
        },
        "train": {
            "caching_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "low_vram_gradient_accumulation_steps": 4,
        },
        "launch_env": {
            "NCCL_P2P_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
        },
        "model": {
            "type": "lumina_2",
            "dtype": "bfloat16",
            "lumina_shift": True,
        },
        "lora_optimizer": OPTIM_ADAMW_OPTIMI,
        "full_optimizer": OPTIM_LUMINA_FULL,
    },
    "hunyuanimage-2.1": {
        "train_kind": "image",
        "modes": {"lora"},
        "required_model_args": [
            "transformer_path",
            "vae_path",
            "text_encoder_path",
            "byt5_path",
        ],
        "optional_model_args": [],
        "min_vram_gb": {"lora": 24},
        "dataset": {"resolutions": [512]},
        "train": {
            "caching_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "low_vram_gradient_accumulation_steps": 4,
        },
        "launch_env": {
            "NCCL_P2P_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
        },
        "model": {
            "type": "hunyuan_image",
            "dtype": "bfloat16",
            "transformer_dtype": "float8",
        },
        "lora_optimizer": OPTIM_ADAMW_OPTIMI,
    },
}


def parse_model_args(items: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid --model-arg '{item}'. Expected key=value.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise SystemExit(f"Invalid --model-arg '{item}'. Expected key=value.")
        parsed[key] = value
    return parsed


def ensure_linux_runtime(action: str) -> None:
    if os.name == "nt":
        raise SystemExit(
            f"{action} must run inside Linux or WSL2. diffusion-pipe is not a native Windows training stack."
        )


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> None:
    preview = shlex.join(command)
    print(f"$ {preview}")
    if dry_run:
        return
    subprocess.run(command, check=True, cwd=cwd, env=env)


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def toml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return toml_quote(value)
    if isinstance(value, list):
        inner = ", ".join(toml_value(item) for item in value)
        return f"[{inner}]"
    if isinstance(value, dict):
        inner = ", ".join(f"{key} = {toml_value(val)}" for key, val in value.items())
        return "{ " + inner + " }"
    raise TypeError(f"Unsupported TOML value type: {type(value)!r}")


def dump_toml_document(document: dict) -> str:
    lines: list[str] = []
    for key, value in document.items():
        if isinstance(value, dict):
            lines.append(f"[{key}]")
            for inner_key, inner_value in value.items():
                lines.append(f"{inner_key} = {toml_value(inner_value)}")
            lines.append("")
        else:
            lines.append(f"{key} = {toml_value(value)}")
    if lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def dump_dataset_toml(
    *,
    resolutions: list[int],
    train_kind: str,
    target_dir: Path,
    control_dir: Path | None,
    dataset_repeat: int,
    caption_prefix: str | None,
) -> str:
    lines = [
        f"resolutions = {toml_value(resolutions)}",
        "",
        "enable_ar_bucket = true",
        "min_ar = 0.5",
        "max_ar = 2.0",
        "num_ar_buckets = 9",
    ]
    if caption_prefix:
        lines.extend(["", f"caption_prefix = {toml_value(caption_prefix)}"])
    lines.extend(["", "[[directory]]", f"path = {toml_quote(str(target_dir))}"])
    if train_kind == "edit":
        lines.append(f"control_path = {toml_quote(str(control_dir))}")
    lines.append(f"num_repeats = {dataset_repeat}")
    return "\n".join(lines) + "\n"


def media_stems(directory: Path) -> set[str]:
    stems: set[str] = set()
    if not directory.exists():
        return stems
    for child in directory.iterdir():
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
            stems.add(child.stem)
    return stems


def validate_edit_pairing(target_dir: Path, control_dir: Path) -> None:
    target_stems = media_stems(target_dir)
    control_stems = media_stems(control_dir)
    if not target_stems or not control_stems:
        return
    if target_stems != control_stems:
        missing_controls = sorted(target_stems - control_stems)
        missing_targets = sorted(control_stems - target_stems)
        message = []
        if missing_controls:
            message.append(f"missing controls for: {', '.join(missing_controls[:10])}")
        if missing_targets:
            message.append(f"missing targets for: {', '.join(missing_targets[:10])}")
        raise SystemExit(
            "Edit dataset target/control stems do not match: " + "; ".join(message)
        )


def dataset_has_media(directory: Path) -> bool:
    return bool(media_stems(directory))


def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_workspace_dirs(workspace_root: Path) -> dict[str, Path]:
    paths = {
        "image_train": workspace_root / "data" / "image" / "train",
        "image_eval": workspace_root / "data" / "image" / "eval",
        "edit_train_target": workspace_root / "data" / "edit" / "train" / "target",
        "edit_train_control": workspace_root / "data" / "edit" / "train" / "control",
        "edit_eval_target": workspace_root / "data" / "edit" / "eval" / "target",
        "edit_eval_control": workspace_root / "data" / "edit" / "eval" / "control",
        "dataset_configs": workspace_root / "configs" / "datasets",
        "train_configs": workspace_root / "configs" / "train",
        "manifests": workspace_root / "configs" / "manifests",
        "runs": workspace_root / "runs",
        "logs": workspace_root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def default_run_name(model: str, train_kind: str, mode: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{model}-{train_kind}-{mode}"


def require_model_args(spec: dict, model_args: dict[str, str]) -> None:
    missing = [key for key in spec["required_model_args"] if key not in model_args]
    if missing:
        raise SystemExit(
            "Missing required --model-arg values: " + ", ".join(missing)
        )


def resolve_pipeline_stages(
    *,
    model: str,
    mode: str,
    num_gpus: int,
    gpu_vram_gb: int,
) -> int:
    if model == "sdxl" and mode == "full" and num_gpus >= 2 and gpu_vram_gb <= 24:
        return 2
    return 1


def build_train_config(
    *,
    args,
    spec: dict,
    run_name: str,
    workspace_dirs: dict[str, Path],
    dataset_config_path: Path,
    eval_dataset_config_path: Path | None,
    model_args: dict[str, str],
) -> tuple[dict, dict[str, str]]:
    low_vram = args.gpu_vram_gb <= 24
    pipeline_stages = resolve_pipeline_stages(
        model=args.model,
        mode=args.mode,
        num_gpus=args.num_gpus,
        gpu_vram_gb=args.gpu_vram_gb,
    )
    if args.num_gpus % pipeline_stages != 0:
        raise SystemExit(
            f"--num-gpus ({args.num_gpus}) must be divisible by pipeline stages ({pipeline_stages})."
        )
    gradient_accumulation_steps = spec["train"]["gradient_accumulation_steps"]
    if low_vram:
        gradient_accumulation_steps = spec["train"].get(
            "low_vram_gradient_accumulation_steps",
            gradient_accumulation_steps,
        )

    train_config = {
        "output_dir": str(workspace_dirs["runs"] / run_name),
        "dataset": str(dataset_config_path),
        "epochs": args.epochs,
        "pipeline_stages": pipeline_stages,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "save_every_n_epochs": min(5, max(1, args.epochs)),
        "caching_batch_size": spec["train"]["caching_batch_size"],
    }
    train_config.update(COMMON_TRAIN_DEFAULTS)
    if eval_dataset_config_path is not None:
        train_config["eval_datasets"] = [
            {"name": "heldout", "config": str(eval_dataset_config_path)}
        ]

    low_vram_blocks_to_swap = spec["train"].get("low_vram_blocks_to_swap")
    if low_vram and low_vram_blocks_to_swap:
        train_config["blocks_to_swap"] = low_vram_blocks_to_swap

    model_config = dict(spec["model"])
    for key in spec["required_model_args"] + spec.get("optional_model_args", []):
        if key in model_args:
            model_config[key] = model_args[key]

    if args.mode == "lora":
        model_config.update(spec.get("lora_model_overrides", {}))
        train_config["adapter"] = {
            "type": "lora",
            "rank": args.rank,
            "dtype": "bfloat16",
        }
        optimizer = dict(spec["lora_optimizer"])
    else:
        model_config.update(spec.get("full_model_overrides", {}))
        optimizer = dict(spec["full_optimizer"])

    train_config["model"] = model_config
    train_config["optimizer"] = optimizer
    return train_config, dict(spec["launch_env"])


def command_preview(command: list[str], env: dict[str, str]) -> str:
    prefix = " ".join(f"{key}={shell_quote(value)}" for key, value in env.items())
    body = shlex.join(command)
    return f"{prefix} {body}".strip()


def cmd_bootstrap(args) -> None:
    ensure_linux_runtime("bootstrap")
    repo_root = Path(args.repo_root).expanduser().resolve()
    python_exe = args.python_bin or sys.executable

    if repo_root.exists():
        run_command(["git", "pull", "--ff-only"], cwd=repo_root, dry_run=args.dry_run)
        run_command(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=repo_root,
            dry_run=args.dry_run,
        )
    else:
        repo_root.parent.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                "git",
                "clone",
                "--recurse-submodules",
                args.clone_url,
                str(repo_root),
            ],
            dry_run=args.dry_run,
        )

    if args.torch_spec:
        run_command(
            [python_exe, "-m", "pip", "install", *shlex.split(args.torch_spec)],
            cwd=repo_root,
            dry_run=args.dry_run,
        )

    if not args.skip_requirements:
        run_command(
            [python_exe, "-m", "pip", "install", "-r", "requirements.txt"],
            cwd=repo_root,
            dry_run=args.dry_run,
        )

    if args.install_flash_attn:
        run_command(
            [python_exe, "-m", "pip", "install", "flash-attn"],
            cwd=repo_root,
            dry_run=args.dry_run,
        )

    if args.install_latest_diffusers:
        run_command(
            [
                python_exe,
                "-m",
                "pip",
                "install",
                "git+https://github.com/huggingface/diffusers",
            ],
            cwd=repo_root,
            dry_run=args.dry_run,
        )


def cmd_prepare(args) -> None:
    ensure_linux_runtime("prepare")
    spec = MODEL_SPECS[args.model]
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1.")
    if args.num_gpus < 1:
        raise SystemExit("--num-gpus must be >= 1.")
    if args.dataset_repeat < 1:
        raise SystemExit("--dataset-repeat must be >= 1.")
    if args.mode == "lora" and args.rank < 1:
        raise SystemExit("--rank must be >= 1 for LoRA runs.")
    if args.train_kind != spec["train_kind"]:
        raise SystemExit(
            f"{args.model} only supports automatic {spec['train_kind']} runs, not {args.train_kind}."
        )
    if args.mode not in spec["modes"]:
        supported = ", ".join(sorted(spec["modes"]))
        raise SystemExit(
            f"{args.model} supports automatic modes: {supported}. Requested: {args.mode}."
        )
    floor = spec["min_vram_gb"][args.mode]
    if args.gpu_vram_gb < floor:
        raise SystemExit(
            f"{args.model} {args.mode} preset requires at least {floor}GB per GPU. "
            f"Requested: {args.gpu_vram_gb}GB."
        )

    model_args = parse_model_args(args.model_arg)
    require_model_args(spec, model_args)

    workspace_root = Path(args.workspace_root).expanduser().resolve()
    workspace_dirs = prepare_workspace_dirs(workspace_root)
    run_name = args.run_name or default_run_name(args.model, args.train_kind, args.mode)

    if args.train_kind == "image":
        train_target = workspace_dirs["image_train"]
        eval_target = workspace_dirs["image_eval"]
        train_control = None
        eval_control = None
    else:
        train_target = workspace_dirs["edit_train_target"]
        train_control = workspace_dirs["edit_train_control"]
        eval_target = workspace_dirs["edit_eval_target"]
        eval_control = workspace_dirs["edit_eval_control"]
        validate_edit_pairing(train_target, train_control)
        validate_edit_pairing(eval_target, eval_control)

    dataset_config_path = workspace_dirs["dataset_configs"] / f"{run_name}_train_dataset.toml"
    eval_dataset_config_path = workspace_dirs["dataset_configs"] / f"{run_name}_eval_dataset.toml"

    dataset_resolutions = (
        [args.resolution] if args.resolution else list(spec["dataset"]["resolutions"])
    )
    caption_prefix = spec["dataset"].get("caption_prefix")

    save_text(
        dataset_config_path,
        dump_dataset_toml(
            resolutions=dataset_resolutions,
            train_kind=args.train_kind,
            target_dir=train_target,
            control_dir=train_control,
            dataset_repeat=args.dataset_repeat,
            caption_prefix=caption_prefix,
        ),
    )

    eval_exists = dataset_has_media(eval_target)
    if args.train_kind == "edit":
        eval_exists = eval_exists and dataset_has_media(eval_control)
    if eval_exists:
        save_text(
            eval_dataset_config_path,
            dump_dataset_toml(
                resolutions=dataset_resolutions,
                train_kind=args.train_kind,
                target_dir=eval_target,
                control_dir=eval_control,
                dataset_repeat=1,
                caption_prefix=caption_prefix,
            ),
        )
    else:
        eval_dataset_config_path = None

    train_config, launch_env = build_train_config(
        args=args,
        spec=spec,
        run_name=run_name,
        workspace_dirs=workspace_dirs,
        dataset_config_path=dataset_config_path,
        eval_dataset_config_path=eval_dataset_config_path,
        model_args=model_args,
    )
    train_config_path = workspace_dirs["train_configs"] / f"{run_name}.toml"
    save_text(train_config_path, dump_toml_document(train_config))

    recommended_command = [
        "deepspeed",
        f"--num_gpus={args.num_gpus}",
        "train.py",
        "--deepspeed",
        "--config",
        str(train_config_path),
    ]

    manifest = {
        "run_name": run_name,
        "workspace_root": str(workspace_root),
        "model": args.model,
        "train_kind": args.train_kind,
        "mode": args.mode,
        "epochs": args.epochs,
        "num_gpus": args.num_gpus,
        "gpu_vram_gb": args.gpu_vram_gb,
        "rank": args.rank if args.mode == "lora" else None,
        "dataset_repeat": args.dataset_repeat,
        "model_args": model_args,
        "train_dataset_config_path": str(dataset_config_path),
        "eval_dataset_config_path": (
            str(eval_dataset_config_path) if eval_dataset_config_path else None
        ),
        "train_config_path": str(train_config_path),
        "launch_env": launch_env,
        "recommended_command": recommended_command,
        "recommended_shell": command_preview(recommended_command, launch_env),
    }
    manifest_path = workspace_dirs["manifests"] / f"{run_name}.json"
    save_json(manifest_path, manifest)

    print(f"Prepared run: {run_name}")
    print(f"Train config: {train_config_path}")
    print(f"Train dataset config: {dataset_config_path}")
    if eval_dataset_config_path:
        print(f"Eval dataset config: {eval_dataset_config_path}")
    else:
        print("Eval dataset config: not created (eval folders are empty)")
    print(f"Manifest: {manifest_path}")
    print("Launch preview:")
    print(manifest["recommended_shell"])


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_launch(args) -> None:
    ensure_linux_runtime("launch")
    repo_root = Path(args.repo_root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    env = os.environ.copy()
    env.update(manifest["launch_env"])

    train_command = list(manifest["recommended_command"])
    extra_flags: list[str] = []
    if args.regenerate_cache:
        extra_flags.append("--regenerate_cache")
    if args.trust_cache:
        extra_flags.append("--trust_cache")
    if args.resume_latest:
        extra_flags.append("--resume_from_checkpoint")
    if args.resume_run:
        extra_flags.extend(["--resume_from_checkpoint", args.resume_run])

    if args.cache_first:
        cache_command = train_command + extra_flags + ["--cache_only"]
        run_command(cache_command, cwd=repo_root, env=env, dry_run=args.dry_run)

    run_command(
        train_command + extra_flags,
        cwd=repo_root,
        env=env,
        dry_run=args.dry_run,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser("bootstrap", help="Clone or update diffusion-pipe and install dependencies.")
    bootstrap.add_argument("--repo-root", required=True, help="Repo path for diffusion-pipe.")
    bootstrap.add_argument(
        "--clone-url",
        default="https://github.com/tdrussell/diffusion-pipe.git",
        help="Git URL to clone when repo-root does not exist.",
    )
    bootstrap.add_argument(
        "--python-bin",
        default=None,
        help="Python executable to use for pip installs. Defaults to the current Python.",
    )
    bootstrap.add_argument(
        "--torch-spec",
        default=None,
        help="Optional torch install spec, for example 'torch torchvision --index-url https://download.pytorch.org/whl/cu128'.",
    )
    bootstrap.add_argument(
        "--skip-requirements",
        action="store_true",
        help="Skip pip install -r requirements.txt.",
    )
    bootstrap.add_argument(
        "--install-flash-attn",
        action="store_true",
        help="Install flash-attn after repo requirements.",
    )
    bootstrap.add_argument(
        "--install-latest-diffusers",
        action="store_true",
        help="Install the latest diffusers from GitHub after repo requirements.",
    )
    bootstrap.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    bootstrap.set_defaults(func=cmd_bootstrap)

    prepare = subparsers.add_parser("prepare", help="Create workspace folders and generate stable configs.")
    prepare.add_argument("--workspace-root", required=True, help="Stable training workspace root.")
    prepare.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    prepare.add_argument("--train-kind", choices=["image", "edit"], required=True)
    prepare.add_argument("--mode", choices=["lora", "full"], required=True)
    prepare.add_argument("--epochs", type=int, required=True)
    prepare.add_argument("--num-gpus", type=int, default=1)
    prepare.add_argument("--gpu-vram-gb", type=int, required=True)
    prepare.add_argument("--rank", type=int, default=32, help="LoRA rank. Ignored for full finetune.")
    prepare.add_argument(
        "--dataset-repeat",
        type=int,
        default=1,
        help="How many repeats each training directory gets per epoch.",
    )
    prepare.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Override the preset resolution with a single square resolution.",
    )
    prepare.add_argument(
        "--run-name",
        default=None,
        help="Optional run name. Defaults to a timestamped model-kind-mode name.",
    )
    prepare.add_argument(
        "--model-arg",
        action="append",
        default=[],
        help="Model-specific key=value setting, repeated as needed.",
    )
    prepare.set_defaults(func=cmd_prepare)

    launch = subparsers.add_parser("launch", help="Run cache generation and launch DeepSpeed training.")
    launch.add_argument("--repo-root", required=True, help="Repo path for diffusion-pipe.")
    launch.add_argument("--manifest", required=True, help="Manifest path generated by the prepare step.")
    launch.add_argument(
        "--cache-first",
        action="store_true",
        help="Run a cache-only pass before the real training launch.",
    )
    launch.add_argument(
        "--regenerate-cache",
        action="store_true",
        help="Force regeneration of the cache files.",
    )
    launch.add_argument(
        "--trust-cache",
        action="store_true",
        help="Skip cache fingerprint checks when loading existing cache metadata.",
    )
    launch.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume from the latest checkpoint for this output directory.",
    )
    launch.add_argument(
        "--resume-run",
        default=None,
        help="Resume from a specific checkpoint folder name.",
    )
    launch.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    launch.set_defaults(func=cmd_launch)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
