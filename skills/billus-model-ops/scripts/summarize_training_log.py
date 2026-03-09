#!/usr/bin/env python3
"""Summarize common training logs into markdown or JSON."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


STEP_PATTERNS = [
    re.compile(r"\bglobal_step\s*[=:]\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\b(?:step|steps|iter|iteration)\s*[=: ]\s*(\d+)\b", re.IGNORECASE),
]

EPOCH_PATTERN = re.compile(r"\bepoch\s*[=: ]\s*([0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE)

METRIC_PATTERNS = {
    "loss": [re.compile(r"\bloss\s*[=:]\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)")],
    "eval_loss": [
        re.compile(r"\beval[_ ]loss\s*[=:]\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)", re.IGNORECASE),
        re.compile(r"\bval[_ ]loss\s*[=:]\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)", re.IGNORECASE),
    ],
    "lr": [re.compile(r"\b(?:lr|learning[_ ]rate)\s*[=:]\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)", re.IGNORECASE)],
    "grad_norm": [re.compile(r"\bgrad[_ ]norm\s*[=:]\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)", re.IGNORECASE)],
    "perplexity": [re.compile(r"\b(?:ppl|perplexity)\s*[=:]\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)", re.IGNORECASE)],
    "accuracy": [re.compile(r"\b(?:acc|accuracy)\s*[=:]\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)", re.IGNORECASE)],
    "f1": [re.compile(r"\bf1\s*[=:]\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)", re.IGNORECASE)],
    "tokens_per_sec": [
        re.compile(r"\b(?:tokens/s|tok/s|toks/s)\s*[=:]?\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)", re.IGNORECASE),
        re.compile(r"\btokens_per_sec\s*[=:]\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)", re.IGNORECASE),
    ],
}

MINIMIZE_METRICS = {"loss", "eval_loss", "perplexity"}
MAXIMIZE_METRICS = {"accuracy", "f1", "tokens_per_sec"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a training log.")
    parser.add_argument("logfile", help="Path to the training log text file")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--tail", type=int, default=5)
    return parser.parse_args()


def maybe_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def extract_step(line: str) -> int | None:
    for pattern in STEP_PATTERNS:
        match = pattern.search(line)
        if match:
            return int(match.group(1))
    return None


def extract_epoch(line: str) -> float | None:
    match = EPOCH_PATTERN.search(line)
    return maybe_float(match.group(1)) if match else None


def extract_metrics(line: str) -> dict[str, float]:
    metrics = {}
    for name, patterns in METRIC_PATTERNS.items():
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                value = maybe_float(match.group(1))
                if value is not None and math.isfinite(value):
                    metrics[name] = value
                    break
    return metrics


def parse_log(path: Path, tail: int) -> dict:
    records = []
    best = {}
    latest = {}
    series = defaultdict(list)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            step = extract_step(line)
            epoch = extract_epoch(line)
            metrics = extract_metrics(line)
            if not metrics:
                continue
            record = {"line": line_no, "step": step, "epoch": epoch, "metrics": metrics}
            records.append(record)
            for metric_name, metric_value in metrics.items():
                latest[metric_name] = metric_value
                series[metric_name].append({"line": line_no, "step": step, "value": metric_value})
                current_best = best.get(metric_name)
                if current_best is None:
                    best[metric_name] = {"line": line_no, "step": step, "value": metric_value}
                elif metric_name in MINIMIZE_METRICS and metric_value < current_best["value"]:
                    best[metric_name] = {"line": line_no, "step": step, "value": metric_value}
                elif metric_name in MAXIMIZE_METRICS and metric_value > current_best["value"]:
                    best[metric_name] = {"line": line_no, "step": step, "value": metric_value}
    return {"logfile": str(path), "records_found": len(records), "latest": latest, "best": best, "recent_records": records[-tail:]}


def fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1:
        return f"{value:.4f}"
    return f"{value:.6g}"


def to_markdown(summary: dict) -> str:
    lines = ["# Training Log Summary", "", f"- File: `{summary['logfile']}`", f"- Parsed metric rows: {summary['records_found']}", "", "## Latest Metrics"]
    for metric_name in sorted(summary["latest"]):
        lines.append(f"- {metric_name}: {fmt(summary['latest'][metric_name])}")
    lines.extend(["", "## Best Metrics"])
    for metric_name in sorted(summary["best"]):
        item = summary["best"][metric_name]
        lines.append(f"- {metric_name}: {fmt(item['value'])} (step={item['step']}, line={item['line']})")
    lines.extend(["", "## Recent Records"])
    for record in summary["recent_records"]:
        parts = [f"line={record['line']}"]
        if record["step"] is not None:
            parts.append(f"step={record['step']}")
        if record["epoch"] is not None:
            parts.append(f"epoch={fmt(record['epoch'])}")
        parts.append(", ".join(f"{k}={fmt(v)}" for k, v in sorted(record["metrics"].items())))
        lines.append(f"- {' | '.join(parts)}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    summary = parse_log(Path(args.logfile), args.tail)
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(to_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
