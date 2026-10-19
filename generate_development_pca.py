"""Plot developmental-stage PCA projections in a single figure.

The projection is calculated from saved neural activity. Each point is
coloured by bilinearly mixing four colours assigned to the corners of the
physical X-Y environment, matching the existing Isomap figures.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from isomap_colors import (
    plot_xy_color_reference,
    position_corner_colors,
    xy_bounds,
)


@dataclass(frozen=True)
class DevelopmentProjection:
    """One developmental condition and its sampled PCA result."""

    label: str
    projection: np.ndarray
    positions: np.ndarray
    sample_indices: np.ndarray
    explained_variance_ratio: np.ndarray
    components: np.ndarray
    mean: np.ndarray


def calculate_projection(
    directory: Path,
    label: str,
    activity_filename: str = "latent_activity.npy",
    slice_start: int | None = None,
    slice_stop: int | None = None,
    max_samples: int = 3_000,
    seed: int = 0,
) -> DevelopmentProjection:
    """Load one activity array, sample it, and project it onto two PCs."""
    activity_path = directory / activity_filename
    positions_path = directory / "positions.npy"
    for path in (activity_path, positions_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")

    activity = np.load(activity_path)
    positions = np.load(positions_path)
    if activity.ndim < 2:
        raise ValueError(f"Activity must have at least 2 dimensions: {activity_path}")
    if positions.ndim < 2 or positions.shape[-1] < 2:
        raise ValueError(
            f"Positions must have x and y in the final dimension: {positions_path}"
        )

    feature_count = activity.shape[-1]
    start = 0 if slice_start is None else slice_start
    stop = feature_count if slice_stop is None else slice_stop
    if not 0 <= start < stop <= feature_count:
        raise ValueError(
            f"Invalid activity slice [{start}:{stop}] for {feature_count} features: "
            f"{activity_path}"
        )
    activity = activity[..., start:stop]

    activity = activity.reshape(-1, activity.shape[-1])
    positions = positions.reshape(-1, positions.shape[-1])
    if len(activity) != len(positions):
        raise ValueError(
            f"Sample counts do not match in {directory}: "
            f"activity={len(activity)}, positions={len(positions)}"
        )
    if len(activity) < 2 or activity.shape[1] < 2:
        raise ValueError(
            f"Two-component PCA requires at least 2 samples and 2 features: "
            f"{activity_path}"
        )
    if not np.all(np.isfinite(activity)):
        raise ValueError(f"Activity contains NaN or infinity: {activity_path}")
    if not np.all(np.isfinite(positions[:, :2])):
        raise ValueError(f"Positions contain NaN or infinity: {positions_path}")

    n_samples = min(max_samples, len(activity))
    # Reinitializing with the same seed for every stage preserves matching
    # sample indices when the saved trajectories have equal lengths.
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(activity), size=n_samples, replace=False)
    activity_sample = activity[sample_indices]

    pca = PCA(n_components=2, svd_solver="randomized", random_state=seed)
    projection = pca.fit_transform(activity_sample)
    explained = pca.explained_variance_ratio_
    print(
        f"{label}: samples={n_samples}, dimensions={activity.shape[1]}, "
        f"explained variance=({explained[0]:.1%}, {explained[1]:.1%})",
        flush=True,
    )

    return DevelopmentProjection(
        label=label,
        projection=projection,
        positions=positions[sample_indices],
        sample_indices=sample_indices,
        explained_variance_ratio=explained,
        components=pca.components_,
        mean=pca.mean_,
    )


def plot_development_projections(
    results: Sequence[DevelopmentProjection],
    output_path: Path,
    point_size: float = 1.0,
    dpi: int = 300,
    title: str | None = None,
) -> Path:
    """Plot the physical-space colour key and all stage-wise PCA results."""
    if not results:
        raise ValueError("At least one developmental condition is required")

    bounds = xy_bounds([result.positions for result in results])
    color_values = [
        position_corner_colors(result.positions, bounds) for result in results
    ]

    n_panels = len(results) + 1
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(2.0 * n_panels, 2.4),
        dpi=dpi,
        squeeze=False,
    )

    plot_xy_color_reference(axes[0, 0], bounds)
    axes[0, 0].tick_params(labelsize=6)
    axes[0, 0].xaxis.label.set_size(7)
    axes[0, 0].yaxis.label.set_size(7)

    for axis, result, colors in zip(axes[0, 1:], results, color_values):
        axis.scatter(
            result.projection[:, 0],
            result.projection[:, 1],
            c=colors,
            s=point_size,
            linewidths=0,
            rasterized=True,
        )
        axis.set_title(result.label, fontsize=10, fontweight="bold")
        axis.set_xlabel(
            f"PC1 ({result.explained_variance_ratio[0]:.1%})", fontsize=7
        )
        axis.set_ylabel(
            f"PC2 ({result.explained_variance_ratio[1]:.1%})", fontsize=7
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[["left", "right", "bottom"]].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    top = 0.78 if title else 0.86
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.13, top=top, wspace=0.28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {output_path}", flush=True)
    return output_path


def save_projection_arrays(
    results: Sequence[DevelopmentProjection], output_path: Path
) -> Path:
    """Save PCA outputs and fitted parameters next to the figure."""
    arrays_path = output_path.with_name(f"{output_path.stem}_data.npz")
    arrays: dict[str, np.ndarray] = {"labels": np.asarray([r.label for r in results])}
    for index, result in enumerate(results):
        arrays[f"projection_{index}"] = result.projection
        arrays[f"positions_{index}"] = result.positions
        arrays[f"sample_indices_{index}"] = result.sample_indices
        arrays[f"explained_variance_ratio_{index}"] = (
            result.explained_variance_ratio
        )
        arrays[f"components_{index}"] = result.components
        arrays[f"mean_{index}"] = result.mean
    np.savez_compressed(arrays_path, **arrays)
    print(f"Saved: {arrays_path}", flush=True)
    return arrays_path


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a developmental-stage PCA figure from activity arrays saved "
            "by RNN_experiment.py. Points use four-corner X-Y colours."
        )
    )
    parser.add_argument(
        "directories",
        nargs="+",
        type=Path,
        help="Activity directories in the order they should be plotted.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Panel labels. Defaults to the input directory names.",
    )
    parser.add_argument(
        "--title",
        help="Optional model title displayed above all developmental stages.",
    )
    parser.add_argument(
        "--activity-file",
        default="latent_activity.npy",
        help="Activity array to project (default: latent_activity.npy).",
    )
    parser.add_argument(
        "--slice-start",
        type=int,
        help="First activity feature to include (inclusive; default: 0).",
    )
    parser.add_argument(
        "--slice-stop",
        type=int,
        help="Last activity feature to include (exclusive; default: all).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper_figures/development_pca.pdf"),
        help="Output figure path (default: paper_figures/development_pca.pdf).",
    )
    parser.add_argument("--samples", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--point-size", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--save-arrays",
        action="store_true",
        help="Save projections, PCA parameters, positions, and sample indices.",
    )
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.directories):
        parser.error("--labels must contain exactly one label per directory")
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if args.slice_start is not None and args.slice_start < 0:
        parser.error("--slice-start must be non-negative")
    if args.slice_stop is not None and args.slice_stop < 1:
        parser.error("--slice-stop must be positive")
    if (
        args.slice_start is not None
        and args.slice_stop is not None
        and args.slice_start >= args.slice_stop
    ):
        parser.error("--slice-start must be smaller than --slice-stop")
    if args.point_size <= 0:
        parser.error("--point-size must be positive")
    if args.dpi < 1:
        parser.error("--dpi must be positive")
    return args


def main() -> None:
    """Calculate and plot all requested developmental conditions."""
    args = parse_args()
    labels = args.labels or [directory.name for directory in args.directories]
    results = [
        calculate_projection(
            directory=directory,
            label=label,
            activity_filename=args.activity_file,
            slice_start=args.slice_start,
            slice_stop=args.slice_stop,
            max_samples=args.samples,
            seed=args.seed,
        )
        for directory, label in zip(args.directories, labels)
    ]
    plot_development_projections(
        results=results,
        output_path=args.output,
        point_size=args.point_size,
        dpi=args.dpi,
        title=args.title,
    )
    if args.save_arrays:
        save_projection_arrays(results, args.output)


if __name__ == "__main__":
    main()
