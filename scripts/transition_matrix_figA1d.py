"""Transition matrix of simulated trajectories drawn like the paper's Figure A1d.

Per seed: the box is split into B x B bins (index = ix*B + iy), positions are
resampled to 1 s, and for every t >= window and k = 1..window seconds
TM[b_t, b_{t-k}] += discount ** ((k-1)/2)   (row = destination, column = start).
Rows are divided by the destination occupancy and smoothed with a Gaussian
(sigma in bins), either over the starting-bin 2D space (reshape to B,B,B,B) or
directly on the B^2 x B^2 matrix. Seeds are averaged over the seeds in which
the destination bin was visited. Variants with and without self-transitions
(b_{t-k} == b_t) are produced.

Outputs in <out-dir>: transition_matrix_figA1d.npz, one PNG per variant and a
two-panel PNG (with / without self-transitions) per smoothing method.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOOTHINGS = ("start2d", "direct")


def seed_matrices(pos, fps, box_length, bins, window, discount, sigma):
    """Return {(self_transitions, smoothing): (B^2, B^2) matrix with NaN unvisited rows}."""
    pos_1s = pos[fps - 1::fps]
    idx = np.clip((pos_1s / box_length * bins).astype(int), 0, bins - 1)
    b = idx[:, 0] * bins + idx[:, 1]

    n = bins * bins
    weights = discount ** ((np.arange(1, window + 1) - 1) / 2)
    out = {}
    for keep_self in (True, False):
        tm = np.zeros((n, n))
        occupancy = np.zeros(n)
        for t in range(window, len(b)):
            occupancy[b[t]] += 1
            for k in range(1, window + 1):
                if keep_self or b[t - k] != b[t]:
                    tm[b[t], b[t - k]] += weights[k - 1]
        visited = occupancy > 0
        tm[visited] /= occupancy[visited, None]

        smoothed = {
            "start2d": gaussian_filter(
                tm.reshape(bins, bins, bins, bins), sigma=sigma, axes=(-2, -1)
            ).reshape(n, n),
            "direct": gaussian_filter(tm, sigma=sigma),
        }
        for name, m in smoothed.items():
            m[~visited] = np.nan
            out[(keep_self, name)] = m
    return out


def draw(ax, tm, bins):
    ax.imshow(tm, cmap="coolwarm", aspect="equal", origin="upper", interpolation="nearest")
    ticks = [1, 20, 40, 60, 80, 100] if bins == 10 else [1, bins * bins]
    ax.set_xticks([t - 1 for t in ticks], labels=ticks)
    ax.set_yticks([t - 1 for t in ticks], labels=ticks)
    ax.set_xlabel("Starting bin")
    ax.set_ylabel("Destination bin")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--stage", default="adult", choices=["crawl", "walk", "run", "adult"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 27)))
    parser.add_argument(
        "--box-length", type=float, default=0.635,
        help="m; the saved positions span 0.005-0.630 m, i.e. a 0.635 m box",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--window", type=int, default=7, help="seconds")
    parser.add_argument("--discount", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=1.0, help="bins")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    per_seed = {}
    for seed in args.seeds:
        exp_dir = next((args.data_root / "box" / args.stage).glob(f"exp_dim*_seed{seed:02d}"))
        pos = np.load(exp_dir / "riab_simulation" / "positions.npy")
        if pos.max() > args.box_length:
            raise ValueError(f"Positions exceed the {args.box_length} m box: {exp_dir}")
        for key, m in seed_matrices(
            pos, args.fps, args.box_length, args.bins, args.window, args.discount, args.sigma
        ).items():
            per_seed.setdefault(key, []).append(m)
    tms = {key: np.nanmean(np.stack(ms), axis=0) for key, ms in per_seed.items()}

    out_dir = args.out_dir or (
        REPO_ROOT / "reports" / "transition_matrix" / f"{args.data_root.name}_{args.stage}_figA1d"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    label = lambda keep_self, smoothing: f"{'self' if keep_self else 'noself'}_{smoothing}"
    np.savez(
        out_dir / "transition_matrix_figA1d.npz",
        **{label(*key): tm for key, tm in tms.items()},
        seeds=np.array(args.seeds), box_length=args.box_length, bins=args.bins,
        window_s=args.window, discount=args.discount, sigma=args.sigma,
    )

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "font.size": 8,
    })
    for key, tm in tms.items():
        fig, ax = plt.subplots(figsize=(2.2, 2.2))
        draw(ax, tm, args.bins)
        fig.tight_layout()
        fig.savefig(out_dir / f"tm_{label(*key)}.png", dpi=300)
        plt.close(fig)
    for smoothing in SMOOTHINGS:
        fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.35))
        for ax, keep_self in zip(axes, (True, False)):
            draw(ax, tms[(keep_self, smoothing)], args.bins)
            ax.set_title("with self-transitions" if keep_self else "without self-transitions")
        fig.tight_layout()
        fig.savefig(out_dir / f"tm_self_vs_noself_{smoothing}.png", dpi=300)
        plt.close(fig)
    print(f"[+] Saved to {out_dir}")


if __name__ == "__main__":
    main()
