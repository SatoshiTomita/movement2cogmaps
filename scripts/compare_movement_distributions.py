"""Overlay movement distributions of two datasets (e.g. without / with obstacles).

Reads movement_distributions.npz written by scripts/movement_distributions.py for
each dataset and draws speed (top row) and rotational speed (bottom row), one
column per stage, with one color per dataset.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = REPO_ROOT / "reports" / "movement_distributions"
STAGES = ["crawl", "walk", "run", "adult"]
# Categorical slots 1 and 2 of the reference palette.
COLORS = ["#2a78d6", "#eb6834"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs=2, default=["data", "data_obst4grid"],
                        help="sub-directories of reports/movement_distributions")
    parser.add_argument("--labels", nargs=2, default=["no obstacles", "4 obstacles"])
    parser.add_argument("--out", type=Path,
                        default=DIST_DIR / "compare_data_vs_data_obst4grid.png")
    args = parser.parse_args()

    dists = [np.load(DIST_DIR / d / "movement_distributions.npz") for d in args.datasets]
    for key in ("speed_edges", "rs_edges"):
        if not np.allclose(dists[0][key], dists[1][key]):
            raise ValueError(f"{key} differ between datasets; recompute with the same bins")

    rows = [("speed", "speed_edges", "Speed (m/s)"), ("rs", "rs_edges", "Rotational speed (rad/s)")]
    fig, axes = plt.subplots(2, len(STAGES), figsize=(13, 5.6), sharey="row")
    for r, (key, edge_key, xlabel) in enumerate(rows):
        edges = dists[0][edge_key]
        centres = (edges[:-1] + edges[1:]) / 2
        for c, stage in enumerate(STAGES):
            ax = axes[r, c]
            for dist, label, color in zip(dists, args.labels, COLORS):
                ax.plot(centres, dist[f"{stage}_{key}_hist"], color=color, lw=2, label=label)
            if r == 0:
                ax.set_title(stage)
            ax.set_xlabel(xlabel)
            if c == 0:
                ax.set_ylabel("Density")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="0.9", lw=0.6)
            ax.set_axisbelow(True)
    axes[0, -1].legend(frameon=False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"[+] Saved {args.out}")


if __name__ == "__main__":
    main()
