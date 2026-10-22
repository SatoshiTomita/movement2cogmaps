"""Speed and rotational-speed distributions of the simulated trajectories.

Uses the same histogram as the rat-data analysis
(real_data_exploration/utils/process_data.py:calculate_hist_occ, bins and ranges
from real_data_exploration/cluster_locomotion/config.yaml) on
<data-root>/box/<stage>/exp_*_seedXX/riab_simulation/{velocities,rot_velocities}.npy,
pooled over the given seeds. Rotational speed is converted from rad per
simulation frame (as stored) to rad/s with the data's FPS, and binned over
[-rs-max, rs-max] rad/s, default [-pi, pi] (the rat analysis bins rad/frame).

Writes <out-dir>/movement_distributions.{npz,png} and summary.csv.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from real_data_exploration.utils.process_data import calculate_hist_occ  # noqa: E402

STAGES = ["crawl", "walk", "run", "adult"]
# Ordinal blue ramp (light -> dark follows crawl -> adult).
STAGE_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


def load_stage(data_root: Path, stage: str, seeds: list[int]):
    speeds, rot_speeds, fps = [], [], None
    for seed in seeds:
        exp_dir = next((data_root / "box" / stage).glob(f"exp_dim*_seed{seed:02d}"))
        sim_dir = exp_dir / "riab_simulation"
        speeds.append(np.linalg.norm(np.load(sim_dir / "velocities.npy"), axis=1))
        fps = yaml.safe_load((exp_dir / "config.yaml").read_text())["FPS"]
        rot_speeds.append(np.load(sim_dir / "rot_velocities.npy") * fps)
    return speeds, rot_speeds, fps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 27)))
    parser.add_argument("--rs-max", type=float, default=np.pi, help="rad/s (default pi)")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(
        (REPO_ROOT / "real_data_exploration/cluster_locomotion/config.yaml").read_text()
    )
    speed_bins, speed_range = cfg["SPEED_BINS"], cfg["SPEED_RANGE"]
    rs_bins, rs_range = cfg["RS_BINS"], [-args.rs_max, args.rs_max]
    speed_edges = np.linspace(*speed_range, speed_bins + 1)
    rs_edges = np.linspace(*rs_range, rs_bins + 1)

    out_dir = args.out_dir or REPO_ROOT / "reports" / "movement_distributions" / args.data_root.name
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = {"speed_edges": speed_edges, "rs_edges": rs_edges, "seeds": np.array(args.seeds)}
    rows = []
    for stage in args.stages:
        speeds, rot_speeds, fps = load_stage(args.data_root, stage, args.seeds)
        arrays[f"{stage}_speed_hist"] = calculate_hist_occ(speeds, speed_bins, speed_range)
        arrays[f"{stage}_rs_hist"] = calculate_hist_occ(rot_speeds, rs_bins, rs_range)
        s, r = np.concatenate(speeds), np.concatenate(rot_speeds)
        rows.append({
            "stage": stage,
            "n_seeds": len(args.seeds),
            "n_samples": len(s),
            "fps": fps,
            "speed_mean_m_s": round(float(s.mean()), 4),
            "speed_median_m_s": round(float(np.median(s)), 4),
            "speed_frac_above_range": round(float((s > speed_range[1]).mean()), 4),
            "abs_rs_mean_rad_s": round(float(np.abs(r).mean()), 4),
            "abs_rs_median_rad_s": round(float(np.median(np.abs(r))), 4),
            "rs_frac_outside_range": round(float((np.abs(r) > args.rs_max).mean()), 4),
        })

    np.savez(out_dir / "movement_distributions.npz", **arrays)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    panels = [
        (axes[0], "speed", speed_edges, "Speed (m/s)"),
        (axes[1], "rs", rs_edges, "Rotational speed (rad/s)"),
    ]
    for ax, key, edges, xlabel in panels:
        centres = (edges[:-1] + edges[1:]) / 2
        for stage, color in zip(args.stages, STAGE_COLORS):
            ax.plot(centres, arrays[f"{stage}_{key}_hist"], color=color, lw=2, label=stage)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, title="stage")
    fig.suptitle(f"{args.data_root.name}: seeds {args.seeds[0]}-{args.seeds[-1]}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "movement_distributions.png", dpi=150)
    plt.close(fig)
    print(f"[+] Saved to {out_dir}")


if __name__ == "__main__":
    main()
