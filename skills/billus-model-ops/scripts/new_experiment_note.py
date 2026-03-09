#!/usr/bin/env python3
"""Generate a concise experiment note for model-training work."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a markdown experiment note.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--goal", default="")
    parser.add_argument("--baseline", default="")
    parser.add_argument("--hypothesis", default="")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--change", action="append", default=[])
    parser.add_argument("--risk", action="append", default=[])
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--command", action="append", default=[])
    parser.add_argument("--out", default="")
    return parser.parse_args()


def bullets(items: list[str], empty_text: str) -> list[str]:
    return [f"- {item}" for item in items] if items else [f"- {empty_text}"]


def render(args: argparse.Namespace) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        f"# {args.title}",
        "",
        f"- Date: {today}",
        f"- Goal: {args.goal or 'TBD'}",
        f"- Baseline: {args.baseline or 'TBD'}",
        "",
        "## Hypothesis",
        args.hypothesis or "TBD",
        "",
        "## Datasets",
        *bullets(args.dataset, "TBD"),
        "",
        "## Candidate Changes",
        *bullets(args.change, "TBD"),
        "",
        "## Main Risks",
        *bullets(args.risk, "TBD"),
        "",
        "## Acceptance Metrics",
        *bullets(args.metric, "TBD"),
        "",
        "## Run Commands",
        *bullets(args.command, "TBD"),
        "",
        "## Results",
        "- Status: planned",
        "- Notes:",
        "",
        "## Decision",
        "- Pending",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    content = render(args)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
    else:
        print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
