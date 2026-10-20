"""Render a saved RatInABox trajectory (riab_simulation/positions.npy) as a GIF.

Unlike scripts/animate_riab_behaviours.py this does not re-simulate: it reads the
positions and the config.yaml copied next to them, so obstacles (OBSTACLES) and
the trajectory are exactly those of the saved experiment. A PNG with the whole
trajectory is written alongside the GIF.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle


def setup_axes(ax, config: dict) -> None:
    env_dim = config["ENV_DIM"]
    ax.set_xlim(0, env_dim)
    ax.set_ylim(0, env_dim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for obstacle in config.get("OBSTACLES") or []:
        cx, cy = obstacle["CENTER"]
        size = obstacle["SIZE"]
        ax.add_patch(
            Rectangle(
                (cx - size / 2, cy - size / 2),
                size,
                size,
                facecolor=obstacle.get("COLOR", "0.35"),
                edgecolor="black",
                linewidth=0.8,
                zorder=3,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_dir", type=Path, help="exp_dim..._seedXX folder")
    parser.add_argument("--frame-step", type=int, default=20, help="samples per GIF frame")
    parser.add_argument("--trail-seconds", type=float, default=20.0)
    parser.add_argument("--gif-fps", type=int, default=20)
    parser.add_argument("--out", type=Path, default=None, help="GIF path")
    args = parser.parse_args()

    config = yaml.safe_load((args.exp_dir / "config.yaml").read_text())
    pos = np.load(args.exp_dir / "riab_simulation" / "positions.npy")
    fps = config["FPS"]
    behaviour = args.exp_dir.parent.name
    label = f"{behaviour} {args.exp_dir.name.rsplit('_', 1)[-1]}"
    out = args.out or Path("reports/riab_gifs") / f"{args.exp_dir.parents[2].name}_{behaviour}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 4))
    setup_axes(ax, config)
    ax.plot(pos[:, 0], pos[:, 1], color="tab:purple", lw=0.5)
    ax.set_title(f"{label}  (0-{len(pos) / fps:.0f} s)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    setup_axes(ax, config)
    history = ax.plot([], [], color="0.8", lw=0.6)[0]
    trail_line = ax.plot([], [], color="tab:purple", lw=1.2)[0]
    head = ax.plot([], [], "o", color="tab:red", ms=4, zorder=4)[0]
    title = ax.set_title(f"{label}   t = 0 s", fontsize=9)
    fig.tight_layout()
    trail = int(args.trail_seconds * fps)

    def update(t):
        history.set_data(pos[:t, 0], pos[:t, 1])
        trail_line.set_data(pos[max(0, t - trail) : t, 0], pos[max(0, t - trail) : t, 1])
        head.set_data(pos[t - 1 : t, 0], pos[t - 1 : t, 1])
        title.set_text(f"{label}   t = {t / fps:.0f} s")
        return history, trail_line, head, title

    frames = range(args.frame_step, len(pos) + 1, args.frame_step)
    FuncAnimation(fig, update, frames=frames, blit=False).save(
        out, writer=PillowWriter(fps=args.gif_fps), dpi=80
    )
    plt.close(fig)
    print(f"[+] Saved {out} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
