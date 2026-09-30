"""Generate and plot Isomap embeddings from saved recurrent activity."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import Isomap

from isomap_colors import (
    plot_xy_color_reference,
    position_corner_colors,
    xy_bounds,
)


def generate_isomap(
    directory: Path,
    activity_filename: str,
    slice_start: int | None,
    slice_stop: int | None,
    output_stem: str,
    max_samples: int,
    max_neighbors: int,
    seed: int,
) -> Path:
    """Create an Isomap plot coloured from the four physical-space corners."""
    activity_path = directory / activity_filename
    positions_path = directory / "positions.npy"

    for path in (activity_path, positions_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")

    activity = np.load(activity_path)
    positions = np.load(positions_path)

    activity = activity[..., slice_start:slice_stop]
    if activity.shape[-1] == 0:
        raise ValueError(
            f"The requested activity slice is empty: "
            f"start={slice_start}, stop={slice_stop}, file={activity_path}"
        )

    activity = activity.reshape(-1, activity.shape[-1])
    positions = positions.reshape(-1, positions.shape[-1])

    if len(activity) != len(positions):
        raise ValueError(
            f"Sample counts do not match in {directory}: "
            f"activity={len(activity)}, positions={len(positions)}"
        )
    if len(activity) < 3:
        raise ValueError(f"Isomap requires at least 3 samples: {directory}")
    if not np.all(np.isfinite(activity)):
        raise ValueError(f"Activity contains NaN or infinity: {activity_path}")
    if positions.shape[1] < 2 or not np.all(np.isfinite(positions[:, :2])):
        raise ValueError(
            f"Positions must contain finite x and y coordinates: {positions_path}"
        )

    n_samples = min(max_samples, len(activity))
    n_neighbors = min(max_neighbors, n_samples - 1)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(activity), size=n_samples, replace=False)

    activity_sample = activity[indices]
    position_sample = positions[indices]

    print(f"Starting: {directory}", flush=True)
    print(
        f"  Isomap input: samples={n_samples}, "
        f"dimensions={activity_sample.shape[1]}, neighbors={n_neighbors}",
        flush=True,
    )

    embedding = Isomap(
        n_components=2,
        n_neighbors=n_neighbors,
        metric="cosine",
        eigen_solver="arpack",
        n_jobs=-1,
    ).fit_transform(activity_sample)

    np.save(directory / f"{output_stem}_embedding.npy", embedding)
    np.save(directory / f"{output_stem}_positions.npy", position_sample)
    np.save(directory / f"{output_stem}_sample_indices.npy", indices)

    bounds = xy_bounds([positions])
    spatial_colors = position_corner_colors(position_sample, bounds)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    plot_xy_color_reference(axes[0], bounds)

    axes[1].scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=spatial_colors,
        s=5,
        alpha=0.9,
        linewidths=0,
        rasterized=True,
    )
    axes[1].set_title("Latent state (Isomap)")
    axes[1].set_xlabel("Isomap 1")
    axes[1].set_ylabel("Isomap 2")

    fig.suptitle(
        f"{directory.parent.name}: "
        f"{activity.shape[-1]} dimensions to 2 dimensions"
    )
    fig.tight_layout()
    output_path = directory / f"{output_stem}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}", flush=True)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Create Isomap embeddings from activity arrays saved by "
            "RNN_experiment.py."
        )
    )
    parser.add_argument(
        "directories",
        nargs="+",
        type=Path,
        help="Activity result directories containing .npy input files.",
    )
    parser.add_argument(
        "--activity-file",
        default="recurrent_activity.npy",
        help=(
            "Activity array to embed. Use recurrent_activity.npy for RNN/RSSM "
            "deterministic-state comparison, or latent_activity.npy for the "
            "complete RSSM state."
        ),
    )
    parser.add_argument(
        "--slice-start",
        type=int,
        default=None,
        help="Optional first activity dimension to include (inclusive).",
    )
    parser.add_argument(
        "--slice-stop",
        type=int,
        default=None,
        help="Optional final activity dimension to include (exclusive).",
    )
    parser.add_argument(
        "--output-stem",
        default="isomap",
        help="Base name for the generated .png and .npy files.",
    )
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--neighbors", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.samples < 3:
        parser.error("--samples must be at least 3")
    if args.neighbors < 1:
        parser.error("--neighbors must be positive")
    if not args.output_stem or Path(args.output_stem).name != args.output_stem:
        parser.error("--output-stem must be a file name without directories")
    return args


def main() -> None:
    """Generate an embedding for every requested analysis directory."""
    args = parse_args()
    for directory in args.directories:
        generate_isomap(
            directory=directory,
            activity_filename=args.activity_file,
            slice_start=args.slice_start,
            slice_stop=args.slice_stop,
            output_stem=args.output_stem,
            max_samples=args.samples,
            max_neighbors=args.neighbors,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
