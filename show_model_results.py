#!/usr/bin/env python3
"""Show the metrics in summary.txt for one model at each development stage."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


STAGES = ("crawl", "walk", "run", "adult")
METRICS = (
    ("place cell候補数", ("Number of place cells", "Selected Place units")),
    ("HD cell候補数", ("Number of HD cells", "Selected HD units")),
    (
        "conjunctive cell候補数",
        ("Number of conjunctive cells", "Place ∧ HD units"),
    ),
    ("pos dec err", ("Position decoding error", "Pos decoding error")),
    ("HD dec err", ("HD decoding error",)),
    ("sRSA", ("sRSA (Spearman corr)",)),
)
CELL_COUNT_METRICS = {
    "place cell候補数",
    "HD cell候補数",
    "conjunctive cell候補数",
}


def canonical_model_name(name: str) -> str:
    """Normalize naming differences introduced by curriculum fine-tuning."""
    for stage in STAGES:
        suffix = f"_{stage}"
        if name.casefold().endswith(suffix):
            name = name[: -len(suffix)]
            break

    # Curriculum steps after crawl are saved with an ``_ft`` marker even
    # though they are stages of the same model. Group those directories with
    # the initial crawl result.
    return re.sub(r"_ft(?=_|$)", "", name, flags=re.IGNORECASE)


def find_summaries(data_dir: Path) -> dict[str, dict[str, list[Path]]]:
    """Return canonical model -> stage -> summary paths."""
    results: dict[str, dict[str, list[Path]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in data_dir.rglob("summary.txt"):
        try:
            relative = path.relative_to(data_dir)
        except ValueError:
            continue

        parts = relative.parts
        # Expected layout: box/<stage>/predictions/.../<model>/<result>/summary.txt
        if len(parts) < 7 or parts[0] != "box" or parts[1] not in STAGES:
            continue
        stage = parts[1]
        model_name = path.parent.parent.name
        results[canonical_model_name(model_name)][stage].append(path)

    for stage_results in results.values():
        for stage, paths in stage_results.items():
            paths.sort(key=lambda path: str(path))
            stage_prefix = f"act_{stage}_"
            matching_activity = [
                path
                for path in paths
                if path.parent.name.casefold().startswith(stage_prefix)
            ]
            if matching_activity:
                stage_results[stage] = matching_activity
    return dict(results)


def choose_model(query: str, model_names: list[str]) -> str | None:
    """Find an exact model name, or ask the user to choose a partial match."""
    normalized = canonical_model_name(query.strip()).casefold()
    exact = [name for name in model_names if name.casefold() == normalized]
    if exact:
        return exact[0]

    matches = [name for name in model_names if normalized in name.casefold()]
    if not matches:
        print(f"モデル '{query}' は見つかりませんでした。", file=sys.stderr)
        return None
    if len(matches) == 1:
        return matches[0]

    if not sys.stdin.isatty():
        print("モデル名に一致する候補が複数あります:", file=sys.stderr)
        for name in matches:
            print(f"  {name}", file=sys.stderr)
        return None

    print("複数のモデルが見つかりました:")
    for index, name in enumerate(matches, start=1):
        print(f"  {index}: {name}")
    while True:
        answer = input("番号を入力してください: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(matches):
            return matches[int(answer) - 1]
        print(f"1〜{len(matches)} の番号を入力してください。")


def latent_activity_dimension(summary_path: Path) -> int | None:
    """Return the number of units that were actually spatially analysed."""
    for filename in ("recurrent_activity.npy", "latent_activity.npy"):
        activity_path = summary_path.with_name(filename)
        if not activity_path.is_file():
            continue
        try:
            activity = np.load(activity_path, mmap_mode="r")
        except (OSError, ValueError):
            continue
        if activity.ndim > 0:
            return int(activity.shape[-1])
    return None


def parse_summary(path: Path) -> dict[str, str]:
    """Read the requested metrics, supporting both summary formats in data/."""
    text = path.read_text(encoding="utf-8", errors="replace")
    dimension_match = re.search(
        r"^Latent space dimension is\s+(\d+)\s+neurons\s*$", text, re.MULTILINE
    )
    # Prefer the saved activity that actually fed the cell analysis. Older
    # results have only the complete latent activity, so retain that fallback.
    total_units = latent_activity_dimension(path)
    if total_units is None and dimension_match:
        total_units = int(dimension_match.group(1))
    values: dict[str, str] = {}
    for display_name, source_names in METRICS:
        value = "—"
        for source_name in source_names:
            match = re.search(
                rf"^{re.escape(source_name)}:\s*(.+?)\s*$", text, re.MULTILINE
            )
            if match:
                # New summaries append "(SEM: ...)"; only the main value is shown.
                value = re.sub(r"\s*\(SEM:.*$", "", match.group(1)).strip()
                break
        if display_name in CELL_COUNT_METRICS and total_units:
            try:
                count = int(value)
            except ValueError:
                pass
            else:
                percentage = count / total_units * 100
                value = f"{count} ({percentage:.1f}%)"
        values[display_name] = value
    return values


def print_results(model: str, stage_results: dict[str, list[Path]]) -> None:
    print(f"\nモデル: {model}")
    for stage in STAGES:
        paths = stage_results.get(stage, [])
        print(f"\n[{stage}]")
        if not paths:
            print("  summary.txt が見つかりません")
            continue

        for index, path in enumerate(paths):
            if len(paths) > 1:
                print(f"  結果: {path.parent.name}")
            metrics = parse_summary(path)
            label_width = max(len(label) for label, _ in METRICS)
            for label, _ in METRICS:
                print(f"  {label:<{label_width}} : {metrics[label]}")
            if index + 1 < len(paths):
                print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="モデルの crawl/walk/run/adult の summary.txt を表示します。"
    )
    parser.add_argument(
        "model", nargs="?", help="モデル名（省略すると対話形式で入力）"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="data ディレクトリ（既定: スクリプトと同じ場所の data）",
    )
    parser.add_argument(
        "--list", action="store_true", help="利用可能なモデル名を一覧表示"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data_dir.is_dir():
        print(f"data ディレクトリが見つかりません: {args.data_dir}", file=sys.stderr)
        return 1

    results = find_summaries(args.data_dir)
    model_names = sorted(results, key=str.casefold)
    if not model_names:
        print(f"summary.txt が見つかりません: {args.data_dir}", file=sys.stderr)
        return 1

    if args.list:
        print("\n".join(model_names))
        return 0

    query = args.model
    if query is None:
        try:
            query = input("モデル名を入力してください: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 130
    if not query:
        print("モデル名が入力されていません。", file=sys.stderr)
        return 2

    model = choose_model(query, model_names)
    if model is None:
        print("利用可能な名前は --list で確認できます。", file=sys.stderr)
        return 2

    print_results(model, results[model])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
