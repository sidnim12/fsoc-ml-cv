from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional


def load_json(path: Path) -> Dict[str, object]:
    """Load one JSON file with a clear error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def safe_float(value: object) -> Optional[float]:
    """Convert numeric-looking values to float while preserving missing values."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: object, digits: int = 3, suffix: str = "") -> str:
    """Format values for a student-readable report."""
    number = safe_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}{suffix}"


def read_final_metrics(path: Path) -> List[Dict[str, str]]:
    """Read final_metrics.csv rows if available."""
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def weak_sequences(rows: Iterable[Mapping[str, object]], lock_threshold: float, accepted_threshold: float) -> List[Mapping[str, object]]:
    """Return sequences that need review based on lock or accepted recall."""
    weak = []
    for row in rows:
        lock = safe_float(row.get("locked_frame_percentage"))
        accepted = safe_float(row.get("accepted_detection_recall"))
        has_target = row.get("accepted_detection_recall") not in (None, "")
        if has_target and ((lock is not None and lock < lock_threshold) or (accepted is not None and accepted < accepted_threshold)):
            weak.append(row)
    return weak


def sequence_name(row: Mapping[str, object]) -> str:
    """Return a compact sequence label from a metrics row."""
    value = str(row.get("sequence_dir") or row.get("outputs_dir") or "sequence")
    return Path(value).name or value


def build_markdown(summary: Mapping[str, object], rows: List[Mapping[str, object]], title: str) -> str:
    """Create a compact Markdown report for final Unity evaluation outputs."""
    aggregate = summary.get("summary", summary)
    if not isinstance(aggregate, Mapping):
        aggregate = {}

    lines = [f"# {title}", ""]
    lines += [
        "## Overall Metrics",
        "",
        f"- Sequences: {aggregate.get('sequence_count', 'n/a')}",
        f"- Successful sequences: {aggregate.get('successful_sequences', 'n/a')}",
        f"- Failed sequences: {aggregate.get('failed_sequences', 'n/a')}",
        f"- Candidate recall: {fmt(aggregate.get('average_candidate_recall'), 6)}",
        f"- Accepted detection recall: {fmt(aggregate.get('average_accepted_detection_recall'), 6)}",
        f"- Filtered MAE: {fmt(aggregate.get('average_filtered_mae_px'), 3, ' px')}",
        f"- Locked frames: {fmt(aggregate.get('average_locked_frame_percentage'), 3, '%')}",
        f"- Effective CPU FPS: {fmt(aggregate.get('average_effective_processing_fps'), 3)}",
        "",
    ]

    if rows:
        lines += [
            "## Sequence Table",
            "",
            "| Sequence | Candidate Recall | Accepted Recall | Filtered MAE | Locked % | Motion Note |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        sequence_name(row),
                        fmt(row.get("candidate_recall"), 6),
                        fmt(row.get("accepted_detection_recall"), 6),
                        fmt(row.get("filtered_mae_px"), 3),
                        fmt(row.get("locked_frame_percentage"), 3),
                        str(row.get("motion_quality_note") or ""),
                    ]
                )
                + " |"
            )
        lines.append("")

        weak = weak_sequences(rows, lock_threshold=50.0, accepted_threshold=0.20)
        lines += ["## Follow-Up List", ""]
        if not weak:
            lines.append("No weak sequences crossed the review thresholds.")
        else:
            for row in weak:
                lines.append(
                    f"- {sequence_name(row)}: lock {fmt(row.get('locked_frame_percentage'), 2, '%')}, "
                    f"accepted recall {fmt(row.get('accepted_detection_recall'), 4)}, "
                    f"MAE {fmt(row.get('filtered_mae_px'), 2, ' px')}"
                )
        lines.append("")

    lines += [
        "## Interpretation",
        "",
        "Candidate recall shows whether OpenCV produced a candidate near the labelled beacon.",
        "Accepted detection recall shows whether the pipeline accepted the correct candidate above confidence threshold.",
        "Filtered MAE shows pixel localization accuracy for accepted/tracked positions.",
        "Lock percentage should be interpreted with the motion-quality note for discontinuous Unity sequences.",
        "",
    ]
    return "\n".join(lines)


def summarize_evaluation(evaluation_root: str | Path, output: str | Path, title: str) -> Path:
    """Summarize one Unity evaluation output folder into a Markdown report."""
    root = Path(evaluation_root)
    summary = load_json(root / "final_summary.json")
    rows = read_final_metrics(root / "final_metrics.csv")
    report = build_markdown(summary, rows, title)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a compact Markdown report from Unity evaluation outputs.")
    parser.add_argument("--evaluation-root", required=True, help="Folder containing final_summary.json and final_metrics.csv.")
    parser.add_argument("--output", default="outputs/reports/unity_evaluation_report.md")
    parser.add_argument("--title", default="Unity Evaluation Report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = summarize_evaluation(args.evaluation_root, args.output, args.title)
    print(f"Report written: {output_path}")


if __name__ == "__main__":
    main()
