#!/usr/bin/env python3
"""Detect likely training-stack components in a model repository."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PATTERNS = {
    "huggingface_trainer": [r"\bTrainer\s*\(", r"TrainingArguments\s*\("],
    "accelerate": [r"\bAccelerator\s*\(", r"accelerate launch"],
    "deepspeed": [r"\bdeepspeed\b", r"\bDeepSpeed\b", r"zero_optimization"],
    "fsdp": [r"\bFSDP\b", r"FullyShardedDataParallel", r"fsdp"],
    "peft": [r"\bLoraConfig\b", r"\bget_peft_model\b", r"\bPeftModel\b", r"\bQLoRA\b"],
    "diffusers": [r"\bDiffusionPipeline\b", r"\bUNet2DConditionModel\b", r"\bAutoencoderKL\b", r"\bControlNetModel\b"],
    "vl_stack": [r"\bvision_tower\b", r"\bprojector\b", r"\bimage_processor\b", r"\bmm_projector\b", r"\bqwen-vl\b", r"\bllava\b", r"\binternvl\b"],
}

FILE_PATTERNS = [
    "*.py",
    "*.yaml",
    "*.yml",
    "*.json",
    "*.toml",
    "*.sh",
    "*.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect likely framework stack in a repo.")
    parser.add_argument("repo_root", help="Repository root to inspect")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser.parse_args()


def scan_files(repo_root: Path) -> tuple[dict[str, list[str]], list[str]]:
    matches = {key: [] for key in PATTERNS}
    candidate_files: list[str] = []
    seen = set()
    for pattern in FILE_PATTERNS:
        for path in repo_root.rglob(pattern):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = str(path.relative_to(repo_root))
            for key, regex_list in PATTERNS.items():
                for regex in regex_list:
                    if re.search(regex, text, re.IGNORECASE):
                        if rel not in matches[key]:
                            matches[key].append(rel)
                        break
            if any(token in rel.lower() for token in ("train", "finetune", "launch", "deepspeed", "accelerate", "config")):
                if rel not in seen:
                    candidate_files.append(rel)
                    seen.add(rel)
    return matches, candidate_files[:20]


def render_text(repo_root: Path, matches: dict[str, list[str]], candidate_files: list[str]) -> str:
    lines = [
        "# Detected Training Stack",
        "",
        f"- Repo: `{repo_root}`",
        "",
        "## Likely Frameworks",
    ]
    any_match = False
    for key in sorted(matches):
        if matches[key]:
            any_match = True
            lines.append(f"- {key}: {', '.join(matches[key][:5])}")
    if not any_match:
        lines.append("- No strong framework signal found")
    lines.extend(["", "## Candidate Entrypoints And Configs"])
    if candidate_files:
        for rel in candidate_files:
            lines.append(f"- {rel}")
    else:
        lines.append("- None found")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    matches, candidate_files = scan_files(repo_root)
    payload = {
        "repo_root": str(repo_root),
        "framework_matches": matches,
        "candidate_files": candidate_files,
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_text(repo_root, matches, candidate_files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
