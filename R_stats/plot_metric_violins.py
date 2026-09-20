#!/usr/bin/env python3
"""Plot SIr, SId, and RVL distributions for R_stats model cases.

Each output contains crawl/walk/run/adult model violins. Experimental data are
shown beside the model for walk/run/adult; no experimental crawl violin is
invented. By default, every case below R_stats is plotted and the figure is
saved as ``violinplot.png`` in that case directory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile

_MPL_CONFIG = Path(tempfile.gettempdir()) / "movement2cogmaps-matplotlib"
_MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
R_STATS = ROOT / "R_stats"
STAGES = ("crawl", "walk", "run", "adult")
METRICS = (
    ("sir", r"SI$_r$ norm.", "#C5268D"),
    ("sid", r"SI$_d$ norm.", "#428808"),
    ("rvl", "RVL norm.", "#428808"),
)
METRIC_FILES = {
    "sir": Path("place/si.npy"),
    "sid": Path("hd/si.npy"),
    "rvl": Path("hd/rvl.npy"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cases",
        nargs="*",
        type=Path,
        help=(
            "R_stats case directory/directories, or parent directories to scan. "
            "Default: R_stats"
        ),
    )
    parser.add_argument(
        "--output-name",
        default="violinplot.png",
        help="Output filename within each case (default: violinplot.png)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Plot the original values instead of source-wise min-max values",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI (default: 300)")
    return parser.parse_args()


def discover_cases(paths: list[Path]) -> list[Path]:
    """Resolve explicit cases and recursively discover exported cases."""
    if not paths:
        paths = [R_STATS]

    cases: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser()
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        if (path / "data/g_model.npy").is_file():
            cases.add(path)
        if not path.is_dir():
            raise ValueError(f"Case or search directory does not exist: {path}")
        cases.update(p.parent.parent.resolve() for p in path.rglob("data/g_model.npy"))

    if not cases:
        raise ValueError("No R_stats cases containing data/g_model.npy were found")
    return sorted(cases, key=str)


def load_vector(path: Path) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"Expected a non-empty, finite 1-D array: {path}")
    return values.astype(float, copy=False)


def split_groups(values: np.ndarray, groups: np.ndarray, path: Path) -> list[np.ndarray]:
    if values.shape != groups.shape:
        raise ValueError(f"Group/value length mismatch: {path}")
    if not np.array_equal(np.unique(groups), [1, 2, 3]):
        raise ValueError(f"Expected groups 1, 2, 3 in {path}")
    return [values[groups == group] for group in (1, 2, 3)]


def replace_path_part(path: Path, old: str, new: str) -> Path:
    parts = list(path.parts)
    try:
        parts[parts.index(old)] = new
    except ValueError as exc:
        raise ValueError(f"Cannot replace path component {old!r} in {path}") from exc
    return Path(*parts)


def crawl_directory(case_dir: Path, manifest: dict) -> Path:
    """Locate the crawl analysis paired with an exported R_stats case."""
    if manifest.get("model_family") == "RNN_0716":
        return (
            ROOT
            / "data/box/crawl/predictions/box_messy/vanilla"
            / "RNN_f1_w9_st10_fss4_do[0,0,0]_lat500_nlsigmoid_hreg0.0_wreg0.0_s01"
            / "act_crawl_epoch1500"
        )

    activity_directories = manifest.get("activity_directories")
    if not isinstance(activity_directories, dict) or "walk" not in activity_directories:
        raise ValueError(f"Missing activity_directories.walk in {case_dir / 'manifest.json'}")
    walk_dir = Path(activity_directories["walk"])

    # Prefer the current checkout if a manifest was moved from another machine.
    try:
        relative = walk_dir.relative_to(ROOT)
    except ValueError:
        marker = Path("data/box")
        parts = walk_dir.parts
        try:
            index = next(i for i in range(len(parts) - 1) if Path(*parts[i : i + 2]) == marker)
        except StopIteration:
            pass
        else:
            walk_dir = ROOT / Path(*parts[index:])

    if "world_model_analysis" in walk_dir.parts:
        crawl_dir = replace_path_part(walk_dir, "walk", "crawl")
        crawl_dir = Path(
            *[
                part.replace("_walk", "_crawl")
                if "_walk" in part
                else part
                for part in crawl_dir.parts
            ]
        )
    else:
        model_name = walk_dir.parent.name.replace("_walk", "_crawl").replace("_ft_", "_")
        activity_name = walk_dir.name.replace("_walk", "_crawl")
        crawl_dir = (
            ROOT
            / "data/box/crawl/predictions/box_messy/vanilla"
            / model_name
            / activity_name
        )
    return crawl_dir


def legacy_case_info(case_dir: Path, units_per_stage: int) -> tuple[Path, np.ndarray, str | None]:
    """Describe older R_stats exports that predate manifest.json."""
    crawl_base = ROOT / "data/box/crawl/predictions/box_messy/vanilla"
    if case_dir == R_STATS:
        directory = (
            crawl_base
            / "RNN_f1_w9_st10_fss4_do[0,0,0]_lat500_nlsigmoid_hreg0.0_wreg0.0_s01"
            / "act_crawl_epoch1500"
        )
        return directory, np.arange(units_per_stage), None

    relative = case_dir.relative_to(R_STATS)
    case_name = case_dir.name
    if relative.parts[0] == "GRU":
        match = re.fullmatch(r"GRU_(\d+)units", case_name)
        if not match:
            raise ValueError(f"Cannot infer the legacy GRU crawl output for {case_dir}")
        directory = crawl_base / f"{case_name}_crawl" / "act_crawl_epoch1500_halfshift"
        return directory, np.arange(units_per_stage), "halfshift"

    if relative.parts[0] == "GRU_0907_100dim":
        directory = crawl_base / "GRU_0907_100dim_crawl" / "act_crawl_epoch1500_halfshift"
        return directory, np.arange(units_per_stage), "halfshift"

    if relative.parts[0] == "RNN":
        match = re.fullmatch(r"RNN_(\d+)units", case_name)
        if not match:
            raise ValueError(f"Cannot infer the legacy RNN crawl output for {case_dir}")
        candidates = sorted(crawl_base.glob(f"{case_name}_*/act_crawl_epoch1500"))
        if len(candidates) != 1:
            raise ValueError(f"Expected one legacy crawl output for {case_dir}; found {candidates}")
        return candidates[0], np.arange(units_per_stage), None

    legacy_rssm = {
        "rssm_0913": crawl_base / "rssm_0913" / "act_crawl_epoch1500_minmax_cells-combined",
        "rssm_d216_s32c32": (
            crawl_base
            / "RSSM_rssm_rnn_d216_s32c32_f1_w9_st10_fss4_do[0.0,0.0,0.0]"
            "_lat216_stoch32_kl1.0_fn0.1_hreg0.0_wreg0.0_s01_cat32"
            / "act_crawl_epoch1500_minmax_cells-combined"
        ),
    }
    family = relative.parts[0]
    if family in legacy_rssm:
        directory = legacy_rssm[family]
        total_units = load_vector(directory / METRIC_FILES["sir"]).size
        if case_name.endswith("_combined"):
            indices = np.arange(total_units)
        elif case_name.endswith("_deterministic"):
            indices = np.arange(units_per_stage)
        elif case_name.endswith("_stochastic"):
            indices = np.arange(total_units - units_per_stage, total_units)
        else:
            raise ValueError(f"Cannot infer the legacy RSSM state selection for {case_dir}")
        return directory, indices, "minmax"

    raise ValueError(
        f"Cannot locate crawl data without {case_dir / 'manifest.json'}. "
        "Re-export this case with R_stats/export_model_stats.py."
    )


def load_case(case_dir: Path) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
    data_dir = case_dir / "data"
    model_groups = load_vector(data_dir / "g_model.npy")
    real_groups = load_vector(data_dir / "g_real.npy")

    model: dict[str, list[np.ndarray]] = {}
    real: dict[str, list[np.ndarray]] = {}
    for metric, _, _ in METRICS:
        model[metric] = split_groups(
            load_vector(data_dir / f"{metric}_model.npy"), model_groups, data_dir
        )
        real[metric] = split_groups(
            load_vector(data_dir / f"{metric}_real.npy"), real_groups, data_dir
        )

    units_per_stage = model["sir"][0].size
    manifest_path = case_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        unit_indices = np.asarray(manifest.get("unit_indices"), dtype=int)
        if unit_indices.ndim != 1 or unit_indices.size != units_per_stage:
            raise ValueError(f"Invalid unit_indices in {manifest_path}")
        crawl_dir = crawl_directory(case_dir, manifest)
        expected_transform = manifest.get("activity_transform")
    else:
        crawl_dir, unit_indices, expected_transform = legacy_case_info(case_dir, units_per_stage)
    transform_path = crawl_dir / "activity_transform.txt"
    if expected_transform is not None:
        if not transform_path.is_file():
            raise ValueError(f"Missing activity transform record: {transform_path}")
        actual_transform = transform_path.read_text(encoding="utf-8").strip()
        if actual_transform != expected_transform:
            raise ValueError(
                f"Activity transform mismatch in {crawl_dir}: "
                f"expected {expected_transform}, got {actual_transform}"
            )

    for metric, metric_path in METRIC_FILES.items():
        all_crawl = load_vector(crawl_dir / metric_path)
        if unit_indices.size and unit_indices.max() >= all_crawl.size:
            raise ValueError(f"unit_indices exceed the crawl array in {crawl_dir / metric_path}")
        model[metric].insert(0, all_crawl[unit_indices])

    return model, real


def minmax_groups(groups: list[np.ndarray]) -> list[np.ndarray]:
    all_values = np.concatenate(groups)
    low = float(all_values.min())
    width = float(all_values.max() - low)
    if width == 0:
        return [np.zeros_like(values) for values in groups]
    return [(values - low) / width for values in groups]


def style_violin(parts: dict, color: str, alpha: float) -> None:
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(alpha)
        body.set_linewidth(0.7)


def plot_case(
    case_dir: Path,
    output_name: str,
    normalize: bool,
    dpi: int,
) -> Path:
    model, real = load_case(case_dir)
    style_path = ROOT / "matplotlib_style.mplstyle"
    with plt.style.context(style_path):
        try:
            font_manager.findfont("Times New Roman", fallback_to_default=False)
        except ValueError:
            plt.rcParams["font.family"] = "DejaVu Serif"

        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.15), dpi=dpi)
        for panel, (ax, (metric, ylabel, color)) in enumerate(zip(axes, METRICS)):
            model_values = minmax_groups(model[metric]) if normalize else model[metric]
            real_values = minmax_groups(real[metric]) if normalize else real[metric]

            model_positions = np.arange(4, dtype=float)
            real_positions = np.arange(1, 4, dtype=float) + 0.16
            model_positions[1:] -= 0.16

            model_parts = ax.violinplot(
                model_values,
                positions=model_positions,
                widths=0.28,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            real_parts = ax.violinplot(
                real_values,
                positions=real_positions,
                widths=0.28,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            style_violin(model_parts, color, 0.82)
            style_violin(real_parts, color, 0.30)

            ax.set_xticks(range(4), STAGES)
            ax.set_xlim(-0.38, 3.45)
            ax.set_ylabel(ylabel if normalize else ylabel.replace(" norm.", ""))
            ax.spines[["right", "top"]].set_visible(False)
            ax.text(
                -0.18,
                1.06,
                chr(ord("a") + panel),
                transform=ax.transAxes,
                fontweight="bold",
                fontsize=11,
                ha="left",
                va="bottom",
            )
            if normalize:
                ax.set_ylim(-0.035, 1.035)
                ax.set_yticks([0.0, 0.5, 1.0])

        source_handles = [
            Line2D([], [], marker="o", linestyle="none", color="black", markersize=7, label="modelled"),
            Line2D([], [], marker="o", linestyle="none", color="0.55", markersize=7, label="experimental"),
        ]
        tuning_handles = [
            Line2D([], [], marker="o", linestyle="none", color="#C5268D", markersize=7, label="positional"),
            Line2D([], [], marker="o", linestyle="none", color="#428808", markersize=7, label="directional"),
        ]
        source_legend = fig.legend(
            handles=source_handles,
            loc="upper right",
            bbox_to_anchor=(0.99, 0.87),
            frameon=False,
            handlelength=0.8,
            handletextpad=0.5,
            fontsize=9,
        )
        fig.add_artist(source_legend)
        fig.legend(
            handles=tuning_handles,
            title="Tuning",
            loc="upper right",
            bbox_to_anchor=(0.99, 0.55),
            frameon=False,
            handlelength=0.8,
            handletextpad=0.5,
            fontsize=9,
            title_fontsize=9,
        )
        fig.subplots_adjust(left=0.07, right=0.82, bottom=0.20, top=0.88, wspace=0.34)

        output = case_dir / output_name
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return output


def main() -> int:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    if Path(args.output_name).name != args.output_name:
        raise ValueError("--output-name must be a filename, not a path")

    cases = discover_cases(args.cases)
    failures: list[tuple[Path, Exception]] = []
    for case in cases:
        try:
            output = plot_case(case, args.output_name, not args.raw, args.dpi)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append((case, exc))
            print(f"ERROR {case}: {exc}", file=sys.stderr)
        else:
            print(output)

    if failures:
        print(
            f"Generated {len(cases) - len(failures)}/{len(cases)} figures; "
            f"{len(failures)} failed.",
            file=sys.stderr,
        )
        return 1
    print(f"Generated {len(cases)} figure(s).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
