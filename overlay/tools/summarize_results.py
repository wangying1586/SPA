from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


METRICS = (
    "specificity",
    "sensitivity",
    "score",
    "accuracy",
    "macro_f1",
    "ece",
    "nll",
    "brier",
)


def infer_metadata(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    text = str(path)
    method = payload.get("method")
    if method is None:
        method = "pafa_spa" if "pafa_spa" in text else "pafa"

    n_cls = payload.get("n_cls")
    if n_cls is None:
        match = re.search(r"([24])class", text)
        n_cls = int(match.group(1)) if match else -1

    test_fold = str(payload.get("test_fold", ""))
    if not test_fold:
        fold_match = re.search(r"fold([0-4])", text)
        test_fold = fold_match.group(1) if fold_match else "official"

    seed = payload.get("seed")
    if seed is None:
        seed_match = re.search(r"seed(\d+)", text)
        seed = int(seed_match.group(1)) if seed_match else -1

    protocol = "5fold" if test_fold in {"0", "1", "2", "3", "4"} else "official_5seed"
    return {
        "method": method,
        "n_cls": int(n_cls),
        "test_fold": test_fold,
        "seed": int(seed),
        "protocol": protocol,
    }


def read_runs(experiments: Path) -> list[dict[str, Any]]:
    chosen: dict[Path, Path] = {}
    for path in experiments.rglob("metrics_eval.json"):
        chosen[path.parent] = path
    for path in experiments.rglob("metrics_best.json"):
        chosen.setdefault(path.parent, path)

    rows = []
    for run_dir, path in sorted(chosen.items()):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = infer_metadata(path, payload)
        row = {**metadata, "run_dir": str(run_dir), "metrics_file": str(path)}
        for metric in METRICS:
            row[metric] = payload.get(metric)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["n_cls"], row["protocol"])].append(row)

    summaries = []
    for (method, n_cls, protocol), group in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "method": method,
            "n_cls": n_cls,
            "protocol": protocol,
            "runs": len(group),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group if row.get(metric) is not None]
            summary[f"{metric}_mean"] = mean(values) if values else None
            summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0 if values else None
        summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser("Aggregate PAFA and PAFA+SPA runs")
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_runs(args.experiments)
    if not rows:
        raise SystemExit(f"No metrics files found below {args.experiments}")
    summaries = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "runs.csv", rows)
    write_csv(args.output_dir / "summary.csv", summaries)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
