"""Render RatInABox trajectories as one GIF per behaviour, with one panel per seed.

The simulation mirrors simulation/riab_simulation/utils.py:run_simulation so the
trajectories match the ones used to build the training data.
"""

import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.animation import FuncAnimation, PillowWriter
from ratinabox.Agent import Agent
from ratinabox.Environment import Environment

CONFIG_PATH = Path(__file__).resolve().parents[1] / "simulation" / "config.yaml"
BEHAVIOURS = ["crawl", "walk", "run", "adult"]


def simulate(config: dict, behaviour: str, seed: int, seconds: float) -> np.ndarray:
    params = config["BEHAVIOURS"][behaviour]
    np.random.seed(seed)
    env = Environment(
        params={"scale": config["ENV_DIM"] - 2 * config["ENV_EPS"], "aspect": 1}
    )
    agent = Agent(
        env,
        params={
            "dt": 1.0 / config["FPS"],
            "speed_mean": params["SPEED_MEAN"],
            "speed_coherence_time": params["SPEED_CT"],
            "rotational_velocity_std": params["ROT_SPEED_STD"] * np.pi,
            "rotational_velocity_coherence_time": params["ROT_SPEED_CT"],
            "thigmotaxis": params["THIGMOTAXIS"],
            "wall_repel_distance": 0.1,
        },
    )
    agent.velocity /= 1e3
    for _ in range(int(seconds * config["FPS"])):
        agent.update(dt=1.0 / config["FPS"])
    return np.array(agent.history["pos"]) + config["ENV_EPS"]


def render(
    behaviour: str,
    seeds: list[int],
    seconds: float,
    frame_step: int,
    trail_seconds: float,
    gif_fps: int,
    out_dir: Path,
) -> Path:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    trajectories = [simulate(config, behaviour, s, seconds) for s in seeds]
    n_steps = len(trajectories[0])
    trail = int(trail_seconds * config["FPS"])
    env_dim = config["ENV_DIM"]

    np.savez(
        out_dir / f"riab_{behaviour}_positions.npz",
        **{f"seed{s:02d}": pos for s, pos in zip(seeds, trajectories)},
    )

    n_cols = min(len(seeds), math.ceil(math.sqrt(len(seeds) * 1.5)))
    n_rows = math.ceil(len(seeds) / n_cols)
    panel = max(1.8, 5.0 / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(panel * n_cols, (panel + 0.1) * n_rows + 0.4), squeeze=False
    )
    history_lines, trail_lines, heads = [], [], []
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, seed in zip(axes.flat, seeds):
        ax.set_axis_on()
        ax.set_xlim(0, env_dim)
        ax.set_ylim(0, env_dim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"seed {seed}", fontsize=8)
        history_lines.append(ax.plot([], [], color="0.8", lw=0.6)[0])
        trail_lines.append(ax.plot([], [], color="tab:purple", lw=1.2)[0])
        heads.append(ax.plot([], [], "o", color="tab:red", ms=3)[0])
    title = fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    frames = range(frame_step, n_steps + 1, frame_step)

    def update(t):
        for pos, hist, tr, head in zip(trajectories, history_lines, trail_lines, heads):
            hist.set_data(pos[:t, 0], pos[:t, 1])
            tr.set_data(pos[max(0, t - trail) : t, 0], pos[max(0, t - trail) : t, 1])
            head.set_data(pos[t - 1 : t, 0], pos[t - 1 : t, 1])
        title.set_text(f"{behaviour}   t = {t / config['FPS']:.1f} s")
        return [*history_lines, *trail_lines, *heads, title]

    out_path = out_dir / f"riab_{behaviour}.gif"
    FuncAnimation(fig, update, frames=frames, blit=False).save(
        out_path, writer=PillowWriter(fps=gif_fps), dpi=60
    )
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behaviours", nargs="+", default=BEHAVIOURS, choices=BEHAVIOURS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 27)))
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--frame-step", type=int, default=5, help="simulation steps per GIF frame")
    parser.add_argument("--trail-seconds", type=float, default=20.0)
    parser.add_argument("--gif-fps", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/riab_gifs"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=len(args.behaviours)) as pool:
        futures = [
            pool.submit(
                render,
                b,
                args.seeds,
                args.seconds,
                args.frame_step,
                args.trail_seconds,
                args.gif_fps,
                args.out_dir,
            )
            for b in args.behaviours
        ]
        for future in futures:
            print(f"[+] Saved {future.result()}")


if __name__ == "__main__":
    main()
