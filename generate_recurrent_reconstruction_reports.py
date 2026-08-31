#!/usr/bin/env python3
"""Generate reconstruction reports for the recurrent baseline models.

The report layout follows ``reports/crssmv4_reconstruction`` but deliberately
omits the bottom absolute-difference row.  Plain RNN/GRU/LSTM models predict
the next observation, while RSSM/MTRSSM additionally reconstruct the current
observation from their posterior state.  The three rows therefore are:

* RNN/GRU/LSTM: input, one-step model output, one-step target;
* RSSM/MTRSSM: input, posterior reconstruction, one-step prior prediction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.data_handler import create_multiple_subsampling, read_video_files_lq


ROOT = Path(__file__).resolve().parent
STAGES = ("crawl", "walk", "run", "adult")
ARCHITECTURES = ("rnn", "gru", "lstm", "rssm", "mtrssm")
DEFAULT_FAMILIES = {
    "rnn": {
        "crawl": "RNN_crawl_0714_",
        "walk": "RNN_0716_walk_",
        "run": "RNN_0716_run_",
        "adult": "RNN_0716_adult_",
    },
    "gru": {stage: f"GRU_0808_50dim_{stage}" for stage in STAGES},
    "lstm": {stage: f"LSTM_LSTM0806_{stage}" for stage in STAGES},
    "rssm": {
        stage: "RSSM_rssm_cat_parammatch_rnn500_" for stage in STAGES
    },
    "mtrssm": {
        stage: "MTRSSM_mtrssm_cat_rnn500match_" for stage in STAGES
    },
}


@dataclass
class ReportFrames:
    first: np.ndarray
    second: np.ndarray
    third: np.ndarray
    mse: float
    row_labels: tuple[str, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=ARCHITECTURES,
        default=list(ARCHITECTURES),
    )
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--seed", type=int, default=22, help="Trajectory seed")
    parser.add_argument("--epoch", type=int, default=1500)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=[0, 1, 5, 10, 20, 40, 80, 150],
        help="Times in seconds in the temporally subsampled trajectory",
    )
    parser.add_argument(
        "--family",
        action="append",
        default=[],
        metavar="ARCH=TEXT",
        help="Override a model-family substring for every requested stage",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "data", help="Repository data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "recurrent_reconstruction",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def family_overrides(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--family must have the form ARCH=TEXT, got {value!r}")
        architecture, text = value.split("=", 1)
        architecture = architecture.casefold()
        if architecture not in ARCHITECTURES or not text:
            raise ValueError(f"Invalid --family value: {value!r}")
        result[architecture] = text
    return result


def find_checkpoint(
    data_dir: Path,
    architecture: str,
    stage: str,
    epoch: int,
    override: str | None,
) -> Path:
    marker = override or DEFAULT_FAMILIES[architecture][stage]
    prediction_root = data_dir / "box" / stage / "predictions" / "box_messy"
    candidates = [
        path
        for path in prediction_root.rglob(f"rnn_epoch{epoch}.pth")
        if marker.casefold() in path.parent.name.casefold()
    ]
    if len(candidates) != 1:
        listing = "\n  ".join(str(path) for path in sorted(candidates)) or "(none)"
        raise FileNotFoundError(
            f"Expected one {architecture}/{stage} checkpoint containing {marker!r}, "
            f"found {len(candidates)}:\n  {listing}"
        )
    return candidates[0]


def minmax(array: np.ndarray) -> np.ndarray:
    minimum = array.min()
    value_range = array.max() - minimum
    if value_range <= np.finfo(array.dtype).eps:
        return np.zeros_like(array)
    return (array - minimum) / value_range


def load_trajectory(data_dir: Path, stage: str, seed: int, stride: int):
    trajectory = data_dir / "box" / stage / f"exp_dim0.635_fps10_s720_seed{seed:02d}"
    video = read_video_files_lq(trajectory / "box_messy", [32, 16]).reshape(-1, 512)
    simulation = trajectory / "riab_simulation"
    velocity = minmax(np.load(simulation / "velocities.npy").astype(np.float32))
    rotation = minmax(
        np.load(simulation / "rot_velocities.npy").astype(np.float32).reshape(-1, 1)
    )

    # Offset zero is one of the stride interleavings used during training.
    observations = create_multiple_subsampling(minmax(video), stride)[0]
    velocity = create_multiple_subsampling(velocity, stride, is_velocity=True)[0]
    rotation = create_multiple_subsampling(rotation, stride, is_velocity=True)[0]
    usable = min(len(observations) - 1, len(velocity), len(rotation))
    current = observations[:usable]
    following = observations[1 : usable + 1]
    action = np.concatenate([velocity[:usable], rotation[:usable]], axis=-1)
    return current, following, action


def prior_stochastic_state(prior) -> torch.Tensor:
    # Use the distribution mean so a report is reproducible and does not
    # depend on stochastic samples drawn while posterior inference is run.
    state = prior.mean
    if state.ndim > 3:
        state = state.flatten(start_dim=2)
    return state.transpose(0, 1)


def infer(
    architecture: str,
    checkpoint: Path,
    current: np.ndarray,
    following: np.ndarray,
    action: np.ndarray,
    device: torch.device,
) -> ReportFrames:
    model = torch.load(checkpoint, map_location=device, weights_only=False)
    model.to(device).eval()
    current_t = torch.from_numpy(current).unsqueeze(0).to(device)
    following_t = torch.from_numpy(following).unsqueeze(0).to(device)
    action_t = torch.from_numpy(action).unsqueeze(0).to(device)

    with torch.inference_mode():
        if architecture in {"rnn", "gru", "lstm"}:
            model_input = torch.cat([current_t, action_t], dim=-1)
            output, _, _ = model(model_input)
            second = output[0].cpu().numpy()
            third = following
            mse = float(np.mean((second - third) ** 2))
            labels = ("Input", "One-step output", "One-step target")
        else:
            if architecture == "mtrssm" and (
                model.decoder.in_features == model.latent_dim_for_action
            ):
                # Older saved models decoded the complete low/high packed
                # state.  The current adapter decodes only the low state, so
                # retain the checkpoint's training-time convention here.
                model._decoder_state = lambda packed_state: packed_state
            output, hidden_all, hidden_last = model.observe(
                action=action_t,
                observation=following_t,
                initial_obs=current_t[:, 0],
            )
            second = output[0].cpu().numpy()

            # Deterministic next states come from the posterior transition, but
            # replacing their stochastic component by the prior mean gives the
            # observation predicted before seeing the next frame.
            if architecture == "rssm":
                determ_dim = model.rssm.determ_dim
                next_determ = torch.cat(
                    [hidden_all[:, 1:, :determ_dim], hidden_last[:, None, :determ_dim]],
                    dim=1,
                )
                prior_latent = torch.cat(
                    [next_determ, prior_stochastic_state(model.last_prior)], dim=-1
                )
                prediction = model.decoder(prior_latent)
            else:
                determ_dim = model.determ_dim
                next_packed = torch.cat(
                    [hidden_all[:, 1:], hidden_last[:, None]], dim=1
                )
                next_determ = torch.cat(
                    [hidden_all[:, 1:, :determ_dim], hidden_last[:, None, :determ_dim]],
                    dim=1,
                )
                prior_parts = [
                    next_determ,
                    prior_stochastic_state(model.last_prior),
                ]
                if model.decoder.in_features == model.latent_dim_for_action:
                    # Preserve the upper-level context used by legacy MTRSSM
                    # decoders; the low stochastic state is the part replaced
                    # by the one-step prior.
                    prior_parts.append(next_packed[..., model.latent_dim :])
                prior_latent = torch.cat(prior_parts, dim=-1)
                prediction = model.decoder(prior_latent)
            third = prediction[0].cpu().numpy()
            mse = float(np.mean((second - current) ** 2))
            labels = ("Input", "Posterior reconstruction", "Prior prediction")

    return ReportFrames(current, second, third, mse, labels)


def time_indices(times: list[float], length: int) -> list[int]:
    indices = [int(round(value)) for value in times]
    invalid = [index for index in indices if index < 0 or index >= length]
    if invalid:
        raise IndexError(f"Time indices outside trajectory of length {length}: {invalid}")
    return indices


def draw_stage_report(
    result: ReportFrames,
    architecture: str,
    stage: str,
    checkpoint: Path,
    seed: int,
    times: list[float],
    output_path: Path,
) -> None:
    indices = time_indices(times, len(result.first))
    figure, axes = plt.subplots(
        3, len(indices), figsize=(2.7 * len(indices), 6.6), squeeze=False
    )
    rows = (result.first, result.second, result.third)
    for row, (images, label) in enumerate(zip(rows, result.row_labels)):
        for column, (seconds, index) in enumerate(zip(times, indices)):
            axis = axes[row, column]
            axis.imshow(np.clip(images[index].reshape(16, 32), 0, 1), cmap="gray", vmin=0, vmax=1)
            axis.set_axis_off()
            if row == 0:
                axis.set_title(f"t={seconds:g}s", fontsize=12)
            if column == 0:
                axis.text(
                    -0.06, 0.5, label, transform=axis.transAxes,
                    ha="right", va="center", rotation=90, fontsize=11,
                )
    figure.suptitle(
        f"{architecture.upper()} reconstruction — {stage} checkpoint, seed {seed} | "
        f"mean MSE={result.mse:.4f}",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.02, 0, 1, 0.91))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def draw_stage_comparison(
    results: dict[str, ReportFrames],
    architecture: str,
    seconds: float,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        3, len(results), figsize=(4.2 * len(results), 7.2), squeeze=False
    )
    for column, (stage, result) in enumerate(results.items()):
        index = time_indices([seconds], len(result.first))[0]
        for row, images in enumerate((result.first, result.second, result.third)):
            axis = axes[row, column]
            axis.imshow(np.clip(images[index].reshape(16, 32), 0, 1), cmap="gray", vmin=0, vmax=1)
            axis.set_axis_off()
            if column == 0:
                axis.text(
                    -0.06, 0.5, result.row_labels[row], transform=axis.transAxes,
                    ha="right", va="center", rotation=90, fontsize=11,
                )
        axes[0, column].set_title(f"{stage}\nmean MSE={result.mse:.4f}", fontsize=13)
    figure.suptitle(
        f"{architecture.upper()} stage comparison at t={seconds:g}s — stage-matched checkpoints and data",
        fontsize=15,
    )
    figure.tight_layout(rect=(0.02, 0, 1, 0.94))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    overrides = family_overrides(args.family)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    for architecture in args.architectures:
        stage_results: dict[str, ReportFrames] = {}
        print(f"[{architecture.upper()}]")
        for stage in args.stages:
            checkpoint = find_checkpoint(
                args.data_dir, architecture, stage, args.epoch, overrides.get(architecture)
            )
            print(f"  {stage}: {checkpoint}", flush=True)
            current, following, action = load_trajectory(
                args.data_dir, stage, args.seed, args.stride
            )
            result = infer(
                architecture, checkpoint, current, following, action, device
            )
            stage_results[stage] = result
            draw_stage_report(
                result,
                architecture,
                stage,
                checkpoint,
                args.seed,
                args.times,
                args.output_dir / f"{architecture}_{stage}_seed{args.seed:02d}.png",
            )
        if stage_results:
            comparison_time = 40.0 if 40.0 in args.times else args.times[0]
            draw_stage_comparison(
                stage_results,
                architecture,
                comparison_time,
                args.output_dir / f"{architecture}_all_stages_seed{args.seed:02d}.png",
            )
    print(f"Saved reports to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
