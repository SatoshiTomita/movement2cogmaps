"""Plot developmental-step Isomap embeddings in a single figure.

The embedding is calculated from neural activity.  As in panel e of
``generate_figure5.ipynb``, each point is coloured by the sum of its physical
x and y coordinates.  No gap/rate-of-change models are included.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from sklearn.manifold import Isomap


@dataclass(frozen=True)
class DevelopmentEmbedding:
    """One developmental condition and its sampled Isomap data."""

    label: str
    embedding: np.ndarray
    positions: np.ndarray
    sample_indices: np.ndarray


def calculate_embedding(
    directory: Path,
    label: str,
    activity_filename: str = "latent_activity.npy",
    max_samples: int = 7_500,
    max_neighbors: int = 100,
    seed: int = 0,
) -> DevelopmentEmbedding:
    """Load and embed one developmental condition."""
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
    if not np.all(np.isfinite(positions[:, :2])):
        raise ValueError(f"Positions contain NaN or infinity: {positions_path}")

    n_samples = min(max_samples, len(activity))
    n_neighbors = min(max_neighbors, n_samples - 1)
    # Reinitializing with the same seed for each condition preserves matching
    # sample indices when the saved trajectories have the same length.
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(activity), size=n_samples, replace=False)

    print(
        f"{label}: samples={n_samples}, dimensions={activity.shape[1]}, "
        f"neighbors={n_neighbors}",
        flush=True,
    )
    embedding = Isomap(
        n_components=2,
        n_neighbors=n_neighbors,
        metric="cosine",
        eigen_solver="arpack",
        n_jobs=-1,
    ).fit_transform(activity[sample_indices])

    return DevelopmentEmbedding(
        label=label,
        embedding=embedding,
        positions=positions[sample_indices],
        sample_indices=sample_indices,
    )


def plot_development_embeddings(
    results: Sequence[DevelopmentEmbedding],
    output_path: Path,
    cmap: str = "coolwarm",
    point_size: float = 1.0,
    dpi: int = 300,
    show_colorbar: bool = False,
) -> Path:
    """Plot every developmental condition in one horizontal figure."""
    if not results:
        raise ValueError("At least one developmental condition is required")

    color_values = [result.positions[:, 0] + result.positions[:, 1] for result in results]
    color_min = min(float(values.min()) for values in color_values)
    color_max = max(float(values.max()) for values in color_values)
    if color_min == color_max:
        color_max = color_min + np.finfo(float).eps
    color_norm = Normalize(vmin=color_min, vmax=color_max)

    fig_width = 1.45 * len(results) + (0.35 if show_colorbar else 0.0)
    fig, axes = plt.subplots(
        1,
        len(results),
        figsize=(fig_width, 1.55),
        dpi=dpi,
        squeeze=False,
    )

    scatter = None
    for axis, result, colors in zip(axes[0], results, color_values):
        scatter = axis.scatter(
            result.embedding[:, 0],
            result.embedding[:, 1],
            c=colors,
            cmap=cmap,
            norm=color_norm,
            s=point_size,
            linewidths=0,
            rasterized=True,
        )
        axis.set_title(result.label)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[["left", "right", "bottom"]].set_visible(False)

    if show_colorbar and scatter is not None:
        colorbar = fig.colorbar(scatter, ax=axes[0].tolist(), fraction=0.025, pad=0.02)
        colorbar.set_label("x + y")

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.82, wspace=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {output_path}", flush=True)
    return output_path


def save_embedding_arrays(
    results: Sequence[DevelopmentEmbedding], output_path: Path
) -> Path:
    """Optionally save the plotted samples next to the figure."""
    arrays_path = output_path.with_name(f"{output_path.stem}_data.npz")
    arrays: dict[str, np.ndarray] = {"labels": np.asarray([r.label for r in results])}
    for index, result in enumerate(results):
        arrays[f"embedding_{index}"] = result.embedding
        arrays[f"positions_{index}"] = result.positions
        arrays[f"sample_indices_{index}"] = result.sample_indices
    np.savez_compressed(arrays_path, **arrays)
    print(f"Saved: {arrays_path}", flush=True)
    return arrays_path


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a Figure-5e-style Isomap figure containing only "
            "developmental conditions. Points are coloured by x + y."
        )
    )
    parser.add_argument(
        "directories",
        nargs="+",
        type=Path,
        help=(
            "Developmental activity directories, in the order to plot "
            "(for example: crawl walk run adult)."
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Panel labels. Defaults to the input directory names.",
    )
    parser.add_argument(
        "--activity-file",
        default="latent_activity.npy",
        help="Activity array to embed (default: latent_activity.npy).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper_figures/development_isomap.pdf"),
        help="Output figure path (default: paper_figures/development_isomap.pdf).",
    )
    parser.add_argument("--samples", type=int, default=7_500)
    parser.add_argument("--neighbors", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cmap", default="coolwarm")
    parser.add_argument("--point-size", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--colorbar",
        action="store_true",
        help="Add a shared x + y colour bar (Figure 5e omits it).",
    )
    parser.add_argument(
        "--save-arrays",
        action="store_true",
        help="Save embeddings, positions, and sampled indices as a compressed NPZ.",
    )
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.directories):
        parser.error("--labels must contain exactly one label per directory")
    if args.samples < 3:
        parser.error("--samples must be at least 3")
    if args.neighbors < 1:
        parser.error("--neighbors must be positive")
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
        calculate_embedding(
            directory=directory,
            label=label,
            activity_filename=args.activity_file,
            max_samples=args.samples,
            max_neighbors=args.neighbors,
            seed=args.seed,
        )
        for directory, label in zip(args.directories, labels)
    ]
    plot_development_embeddings(
        results=results,
        output_path=args.output,
        cmap=args.cmap,
        point_size=args.point_size,
        dpi=args.dpi,
        show_colorbar=args.colorbar,
    )
    if args.save_arrays:
        save_embedding_arrays(results, args.output)


if __name__ == "__main__":
    main()
