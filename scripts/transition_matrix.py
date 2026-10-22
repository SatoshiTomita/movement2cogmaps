"""Transition matrix of simulated trajectories, as in the rat-data analysis.

Uses real_data_exploration/utils/transition_matrix_helper.py (generate_tm_loop,
reorder/fold) with the parameters of
real_data_exploration/cluster_locomotion/config.yaml (B x B bins, discount DF
over a THRESHOLD_SEC window, Gaussian smoothing SIGMA_SMOOTHING). The window and
the discount are converted to frames with the simulation FPS, so the time
constants match the rat analysis. Positions come from
<data-root>/box/<stage>/exp_*_seedXX/riab_simulation/positions.npy.

tm[x0, y0] is the B x B distribution of the bins visited during the preceding
THRESHOLD_SEC window (discounted), given the current bin (x0, y0); each
non-empty tm[x0, y0] sums to 1. Writes <out-dir>/transition_matrix.{npz,png}.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from real_data_exploration.utils.transition_matrix_helper import (  # noqa: E402
    fold_transition_matrix,
    generate_tm_loop,
    reord_folded_transition_matrix,
    reorder_transition_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--stage", default="adult", choices=["crawl", "walk", "run", "adult"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 27)))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(
        (REPO_ROOT / "real_data_exploration/cluster_locomotion/config.yaml").read_text()
    )
    bins = cfg["B"]
    half_idx = bins**2 // 2

    xs, ys = [], []
    for seed in args.seeds:
        exp_dir = next((args.data_root / "box" / args.stage).glob(f"exp_dim*_seed{seed:02d}"))
        pos = np.load(exp_dir / "riab_simulation" / "positions.npy")
        xs.append(pos[:, 0])
        ys.append(pos[:, 1])
        sim_cfg = yaml.safe_load((exp_dir / "config.yaml").read_text())
    # BOX_DIM rather than the data's ENV_DIM: the config.yaml copies in data/box
    # carry a stale ENV_DIM (0.625) although the trajectories span 0-0.635 m.
    fps, box_length = sim_cfg["FPS"], cfg["BOX_DIM"]
    if max(max(x.max(), y.max()) for x, y in zip(xs, ys)) > box_length:
        raise ValueError(f"Positions exceed the {box_length} m box")

    window = int(fps * cfg["THRESHOLD_SEC"])
    df_damp = 1.0 / (2 * fps)
    tm, tm_occ = generate_tm_loop(
        xs, ys, box_length, bins,
        cfg["DF"], df_damp, window, cfg["ATOL"],
        cfg["SIGMA_SMOOTHING"], subsample=1, plot=False,
    )
    tm_reord = reorder_transition_matrix(tm, bins)
    tm_reord_quarter, _, _, _ = fold_transition_matrix(tm, tm_reord, bins, half_idx)
    tm_fold = reord_folded_transition_matrix(tm_reord_quarter, bins)

    out_dir = args.out_dir or (
        REPO_ROOT / "reports" / "transition_matrix" / f"{args.data_root.name}_{args.stage}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "transition_matrix.npz",
        tm=tm, tm_occ=tm_occ, tm_reord=tm_reord, tm_fold=tm_fold,
        seeds=np.array(args.seeds), fps=fps, box_length=box_length, bins=bins,
        window_frames=window, discount=cfg["DF"], df_damp=df_damp,
        sigma=cfg["SIGMA_SMOOTHING"],
    )

    # Left: reordered matrix; block (i, j) is tm[i, j], i.e. the past-bin map
    # (rows = past x, columns = past y) for current bin x = i, y = j.
    # Right: distributions for a corner, an edge and a central current bin.
    examples = [(0, 0), (0, bins // 2), (bins // 2, bins // 2)]
    fig = plt.figure(figsize=(12, 4.2))
    ax_full = fig.add_axes([0.05, 0.12, 0.27, 0.76])
    im = ax_full.imshow(tm_reord, cmap="Blues", interpolation="nearest")
    ticks = np.arange(bins) * bins + bins / 2 - 0.5
    ax_full.set_xticks(ticks, labels=range(bins))
    ax_full.set_yticks(ticks, labels=range(bins))
    for edge in np.arange(1, bins) * bins - 0.5:
        ax_full.axhline(edge, color="white", lw=0.5)
        ax_full.axvline(edge, color="white", lw=0.5)
    ax_full.set_title("All current bins (block = past-bin map)", fontsize=9)
    ax_full.set_xlabel("current bin y")
    ax_full.set_ylabel("current bin x")
    fig.colorbar(im, ax=ax_full, fraction=0.046, pad=0.02, label="probability")
    extent = (0, box_length * 100, 0, box_length * 100)
    for k, (x0, y0) in enumerate(examples):
        ax = fig.add_axes([0.45 + k * 0.185, 0.12, 0.165, 0.76])
        ax.imshow(tm[x0, y0].T, origin="lower", cmap="Blues", extent=extent)
        cell = box_length * 100 / bins
        ax.add_patch(plt.Rectangle(
            (x0 * cell, y0 * cell), cell, cell, fill=False, edgecolor="#eb6834", lw=1.5
        ))
        ax.set_title(f"current bin ({x0}, {y0})", fontsize=9)
        ax.set_xlabel("x (cm)")
        if k == 0:
            ax.set_ylabel("y (cm)")
        else:
            ax.set_yticklabels([])
    fig.suptitle(
        f"{args.data_root.name} {args.stage}, seeds {args.seeds[0]}-{args.seeds[-1]}: "
        f"{bins}x{bins} bins, {cfg['THRESHOLD_SEC']} s window, DF {cfg['DF']}",
        fontsize=10,
    )
    fig.savefig(out_dir / "transition_matrix.png", dpi=150)
    plt.close(fig)
    print(f"[+] Saved to {out_dir}")


if __name__ == "__main__":
    main()
