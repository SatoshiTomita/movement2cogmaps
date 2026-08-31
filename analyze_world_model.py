#!/usr/bin/env python3
"""Run spatial-unit analyses for staged world-model checkpoints.

The training entry point in :mod:`maincrssmV4` stores Lightning state-dict
checkpoints under ``models/params/<model>/alpha:<a>/seed:<s>/<stage>``.  The
older recurrent analysis code expects complete pickled models instead.  This
module bridges the two layouts: it rebuilds the world model from its Hydra
configuration, extracts its latent activity, and feeds that activity to the
existing place/HD-cell analysis implementation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from architectures.world import CoarseWorldModel, WorldModel
from utils.activiter import RNNActiviter
from utils.config import CRSSMConfig, CRSSMV4Config, load_config
from utils.trainer import RNNTrainer
from utils.utils import torch_fix_seed


STAGE_DIRS = {
    "crawl": "01_crawl",
    "walk": "02_walk",
    "run": "03_run",
    "adult": "04_adult",
}

CELL_ACTIVITY_DESCRIPTIONS = {
    "lower_combined": (
        "world-model lower (precise) deterministic and posterior stochastic "
        "states"
    ),
    "upper_combined": (
        "world-model upper (coarse) deterministic and posterior stochastic "
        "states"
    ),
    "deterministic": (
        "world-model lower (precise) and upper (coarse) deterministic states"
    ),
    "stochastic": (
        "world-model lower (precise) and upper (coarse) posterior stochastic "
        "states"
    ),
    # Retain the previous whole-latent option for reproducibility.
    "combined": (
        "world-model lower/upper deterministic and posterior stochastic states"
    ),
}


def parse_stages(value: str) -> list[str]:
    stages = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [stage for stage in stages if stage not in STAGE_DIRS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown stages: {', '.join(unknown)}")
    if not stages:
        raise argparse.ArgumentTypeError("at least one stage is required")
    return stages


def resolve_device(value: str) -> torch.device:
    if value.casefold() == "cpu":
        return torch.device("cpu")
    if not value.isdigit():
        raise ValueError("--device must be 'cpu' or a non-negative CUDA index")
    if not torch.cuda.is_available():
        print("[!] CUDA is unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(f"cuda:{value}")


def make_loader_args(args: argparse.Namespace, stage: str) -> SimpleNamespace:
    """Build the subset of RNNTrainer arguments used by its data pipeline."""
    return SimpleNamespace(
        architecture="crssmv4",
        behaviour=stage,
        env=args.env,
        env_dim=args.env_dim,
        moredata=None,
        activity_only=True,
        epochs=2,
        stride=args.stride,
        bptt_steps=args.bptt_steps,
        n_future_pred=1,
        n_gridcells=0,
        num_workers=args.num_workers,
        name_prefix=None,
        pretrained_model_folder=None,
        pretrained_behav=None,
        reset_hidden_at=None,
        dropouts=[0, 0, 0],
        latent_dim=1,
        stoch_dim=1,
        kl_scale=1.0,
        free_nats=0.0,
        hidden_reg=0.0,
        weights_reg=0.0,
        seed=args.seed,
        stoch_dist="normal",
    )


def load_activity_arrays(
    args: argparse.Namespace, stage: str
) -> tuple[SimpleNamespace, tuple[np.ndarray, ...]]:
    loader_args = make_loader_args(args, stage)
    factory = RNNTrainer(
        loader_args,
        args.data_root,
        torch.device("cpu"),
        create_dir=False,
    )
    videos, thetas, positions, velocities, rotational_velocities = (
        factory.load_simulations()
    )
    videos, velocities, rotational_velocities = factory.preprocess_data(
        videos, velocities, rotational_velocities
    )
    split = factory.combine_videos(
        videos, velocities, rotational_velocities, positions, thetas
    )
    # The final five arrays are the activity split.
    return loader_args, tuple(split[10:15])


def build_world(
    config_node: DictConfig,
    obs_shape: tuple[int, ...],
    action_dim: int,
) -> WorldModel:
    config_node = config_node.copy()
    # ``callbacks`` configures the Lightning trainer and is not a field of
    # ExperimentConfig (the training entry point removes it for the same reason).
    config_node.pop("callbacks", None)
    config_node.obs_shape = obs_shape
    config_node.action_dim = action_dim
    OmegaConf.resolve(config_node)
    config = instantiate(config_node)
    config.world.dict2dc()
    if isinstance(config.world.dynamics_cfg, (CRSSMV4Config, CRSSMConfig)):
        return CoarseWorldModel(config.world)
    return WorldModel(config.world)


def checkpoint_path(args: argparse.Namespace, stage: str) -> Path:
    stage_dir = (
        Path("models/params")
        / args.model_name
        / f"alpha:{args.alpha}"
        / f"seed:{args.seed}"
        / STAGE_DIRS[stage]
    )
    filename = "last.ckpt" if args.load_last else "CoarseWorldModel.ckpt"
    path = stage_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path


def load_checkpoint(world: WorldModel, path: Path, device: torch.device) -> None:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = saved.get("state_dict", saved)
    # Older checkpoints may contain a regenerated action-transform kernel.
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if key != "action_transform.k" and "codebook.mask" not in key
    }
    world.load_state_dict(state_dict, strict=True)
    world.to(device).eval()


def transform_activity(activity: np.ndarray, method: str) -> np.ndarray:
    if method == "identity":
        return activity
    if method == "softplus":
        return np.logaddexp(0, activity)
    if method == "halfshift":
        return 0.5 * activity + 0.5
    if method == "minmax":
        minimum = activity.min(axis=(0, 1), keepdims=True)
        span = activity.max(axis=(0, 1), keepdims=True) - minimum
        return np.divide(
            activity - minimum,
            span,
            out=np.zeros_like(activity),
            where=span > np.finfo(activity.dtype).eps,
        )
    raise ValueError(f"unknown activity transform: {method}")


def select_cell_activity(states, source: str) -> torch.Tensor:
    """Select the CRSSMV4 state components used for spatial-cell analysis."""
    lower_deterministic = states.determ
    lower_stochastic = states.posterior.stoch

    if source == "lower_combined":
        return torch.cat([lower_deterministic, lower_stochastic], dim=-1)
    if source == "combined":
        return states.latent_states

    if not hasattr(states, "coarse"):
        raise ValueError(
            f"--cell-activity-source {source!r} requires a hierarchical "
            "CRSSM/CRSSMV4 checkpoint"
        )
    upper_deterministic = states.coarse
    upper_posterior = getattr(states, "c_posterior", None)
    if source == "deterministic":
        return torch.cat(
            [lower_deterministic, upper_deterministic], dim=-1
        )
    if upper_posterior is None:
        raise ValueError(
            f"--cell-activity-source {source!r} requires an upper posterior "
            "stochastic state; use a CRSSMV4 checkpoint"
        )
    upper_stochastic = upper_posterior.stoch
    if source == "upper_combined":
        return torch.cat([upper_deterministic, upper_stochastic], dim=-1)
    if source == "stochastic":
        return torch.cat([lower_stochastic, upper_stochastic], dim=-1)
    raise ValueError(f"unknown cell activity source: {source}")


@torch.no_grad()
def extract_activity(
    world: WorldModel,
    arrays: tuple[np.ndarray, ...],
    obs_shape: tuple[int, ...],
    device: torch.device,
    batch_size: int,
    deterministic: bool,
    cell_activity_source: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    videos, velocities, rotational_velocities, positions, thetas = arrays
    # Velocity preprocessing can remove its final sample.  Use the longest
    # common prefix that still has a next-frame observation target.
    n_steps = min(
        videos.shape[1] - 1,
        velocities.shape[1],
        rotational_velocities.shape[1],
        positions.shape[1],
        thetas.shape[1],
    )
    videos_in = videos[:, :n_steps]
    videos_target = videos[:, 1:n_steps + 1]
    velocities = velocities[:, :n_steps]
    rotational_velocities = rotational_velocities[:, :n_steps]
    positions = positions[:, :n_steps]
    thetas = thetas[:, :n_steps]

    latent_batches: list[np.ndarray] = []
    recurrent_batches: list[np.ndarray] = []
    cell_activity_batches: list[np.ndarray] = []
    squared_error = 0.0
    n_pixels = 0

    for start in range(0, len(videos_in), batch_size):
        stop = min(start + batch_size, len(videos_in))
        obs = torch.from_numpy(videos_in[start:stop]).float()
        obs = obs.reshape(stop - start, obs.shape[1], *obs_shape)
        obs = obs.transpose(0, 1).to(device)
        target = torch.from_numpy(videos_target[start:stop]).float()
        target = target.reshape(stop - start, target.shape[1], *obs_shape)
        target = target.transpose(0, 1).to(device)

        action_np = np.concatenate(
            [velocities[start:stop], rotational_velocities[start:stop]], axis=-1
        )
        action = torch.from_numpy(action_np).float().transpose(0, 1).to(device)
        action = world.action_transform(action)

        embedded = world.obs_encoder(obs)
        world.dynamics.init_latent(embedded.shape[1], embedded[0])
        states, _ = world.dynamics(action, embedded, deterministic=deterministic)
        latent = states.latent_states
        recurrent = (
            torch.cat([states.determ, states.coarse], dim=-1)
            if hasattr(states, "coarse")
            else states.determ
        )
        cell_activity = select_cell_activity(states, cell_activity_source)

        prediction, _ = world._decode_obs(latent, embedded)
        if prediction is not None:
            squared_error += torch.sum((prediction - target) ** 2).item()
            n_pixels += target.numel()

        latent_batches.append(latent.transpose(0, 1).cpu().numpy())
        recurrent_batches.append(recurrent.transpose(0, 1).cpu().numpy())
        cell_activity_batches.append(
            cell_activity.transpose(0, 1).cpu().numpy()
        )
        print(f"    trajectories {start + 1}-{stop}/{len(videos_in)}")

    reconstruction_mse = squared_error / n_pixels if n_pixels else float("nan")
    return (
        np.concatenate(latent_batches, axis=0),
        np.concatenate(recurrent_batches, axis=0),
        np.concatenate(cell_activity_batches, axis=0),
        positions,
        thetas,
        reconstruction_mse,
    )


def output_dir(args: argparse.Namespace, stage: str) -> Path:
    result_name = f"act_{stage}_checkpoint"
    if (
        args.cell_activity_source != "deterministic"
        or not args.deterministic_latents
    ):
        latent_mode = "argmax" if args.deterministic_latents else "sampled"
        result_name += (
            f"_cells-{args.cell_activity_source}_latents-{latent_mode}"
        )
    return (
        Path(args.data_root)
        / "box"
        / stage
        / "predictions"
        / args.env
        / "world_model_analysis"
        / f"{args.model_name}_{stage}"
        / result_name
    )


def run_spatial_analysis(
    args: argparse.Namespace,
    loader_args: SimpleNamespace,
    stage: str,
    latent: np.ndarray,
    recurrent: np.ndarray,
    cell_activity: np.ndarray,
    positions: np.ndarray,
    thetas: np.ndarray,
    reconstruction_mse: float,
) -> Path:
    result_dir = output_dir(args, stage)
    result_dir.mkdir(parents=True, exist_ok=True)

    latent = transform_activity(latent, args.activity_transform)
    recurrent = transform_activity(recurrent, args.activity_transform)
    cell_activity = transform_activity(
        cell_activity, args.activity_transform
    )
    cell_activity_description = CELL_ACTIVITY_DESCRIPTIONS[
        args.cell_activity_source
    ]
    np.save(result_dir / "latent_activity.npy", latent)
    np.save(result_dir / "recurrent_activity.npy", recurrent)
    np.save(result_dir / "cell_activity.npy", cell_activity)
    np.save(result_dir / "positions.npy", positions)
    np.save(result_dir / "thetas.npy", thetas)
    (result_dir / "cell_activity_dimension.txt").write_text(
        f"{cell_activity.shape[-1]}\n", encoding="utf-8"
    )
    (result_dir / "activity_transform.txt").write_text(
        f"{args.activity_transform}\n", encoding="utf-8"
    )
    (result_dir / "cell_activity_source.txt").write_text(
        f"{cell_activity_description}\n", encoding="utf-8"
    )

    analysis_args = SimpleNamespace(
        behaviour=stage,
        behaviour_act=stage,
        activity_transform=args.activity_transform,
        env_dim=args.env_dim,
        ratemap_norm=args.ratemap_norm,
        wandb=False,
        seeds_act=loader_args.seeds_act,
    )
    activiter = RNNActiviter(
        analysis_args,
        args.data_root,
        torch.device("cpu"),
        args.model_name,
        str(result_dir),
    )
    activiter.cell_activity_description = cell_activity_description

    (
        cell_activity_half1,
        positions_half1,
        thetas_half1,
        cell_activity_half2,
        positions_half2,
        thetas_half2,
    ) = activiter.split_data(cell_activity, positions, thetas)
    latent_half1, _, _, latent_half2, _, _ = activiter.split_data(
        latent, positions, thetas
    )

    print("  Calculating sRSA")
    srsa = activiter.calculate_sRSA(latent, positions)
    flat_cell_activity = cell_activity.reshape(-1, cell_activity.shape[-1])
    flat_positions = positions.reshape(-1, positions.shape[-1])
    flat_thetas = thetas.reshape(-1, thetas.shape[-1]).squeeze()

    print("  Calculating place-cell metrics")
    (
        rate_maps,
        si_r,
        place_cells,
        n_fields,
        rm_stability,
        single_field_dim,
        rm_vs_hd,
        rm_vs_hd_stability,
    ) = activiter.rnn_place_activity(
        flat_cell_activity,
        cell_activity_half1,
        cell_activity_half2,
        flat_positions,
        positions_half1,
        positions_half2,
        flat_thetas,
    )

    print("  Calculating head-direction metrics")
    (
        polar_maps,
        si_d,
        rvl,
        rvangle,
        hd_cells,
        pm_stability,
        pm_vs_place,
        pm_vs_place_stability,
    ) = activiter.rnn_hd_activity(
        flat_cell_activity,
        cell_activity_half1,
        cell_activity_half2,
        flat_thetas,
        thetas_half1,
        thetas_half2,
        flat_positions,
    )

    pos_error, theta_error = activiter.pos_hd_decoding(
        latent_half1,
        latent_half2,
        positions_half1,
        positions_half2,
        thetas_half1,
        thetas_half2,
    )
    if args.skip_plots:
        conjunctive_cells = np.intersect1d(place_cells, hd_cells)
        place_cells = np.setdiff1d(place_cells, hd_cells)
        hd_cells = np.setdiff1d(hd_cells, conjunctive_cells)
        np.save(result_dir / "indices_place_cells.npy", place_cells)
        np.save(result_dir / "indices_hd_cells.npy", hd_cells)
        np.save(result_dir / "indices_conjunctive_cells.npy", conjunctive_cells)
    else:
        place_cells, hd_cells, conjunctive_cells = (
            activiter.selected_units_analysis(
                place_cells,
                hd_cells,
                rate_maps,
                rm_vs_hd,
                rm_vs_hd_stability,
                polar_maps,
                pm_vs_place,
                pm_vs_place_stability,
            )
        )
        activiter.save_place_plots(
            rate_maps, si_r, place_cells, hd_cells, conjunctive_cells
        )
        activiter.save_hd_plots(
            polar_maps,
            si_d,
            rvl,
            rvangle,
            place_cells,
            hd_cells,
            conjunctive_cells,
        )
    activiter.save_summary(
        {"Reconstruction MSE": reconstruction_mse},
        pos_error,
        theta_error,
        srsa,
        place_cells,
        n_fields,
        rm_stability,
        single_field_dim,
        rm_vs_hd_stability,
        hd_cells,
        pm_stability,
        pm_vs_place_stability,
        conjunctive_cells,
    )
    return result_dir


def main(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    config_path = Path("models/cfg") / f"{args.model_name}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"model config not found: {config_path}")
    base_config = load_config(str(config_path))
    base_config.alpha = args.alpha
    base_config.seed = args.seed
    base_config.device = 0 if device.type == "cuda" else None
    torch_fix_seed(args.seed)

    data_cfg = base_config.datamodule
    args.data_root = args.data_root or data_cfg.data_root
    args.env = args.env or data_cfg.env
    args.env_dim = args.env_dim if args.env_dim is not None else data_cfg.env_dim
    args.stride = args.stride if args.stride is not None else data_cfg.stride
    args.bptt_steps = (
        args.bptt_steps if args.bptt_steps is not None else data_cfg.chunk_size
    )
    args.num_workers = (
        args.num_workers if args.num_workers is not None else data_cfg.num_workers
    )

    for stage in args.stages:
        print(f"\n[+] Analysing {args.model_name}: {stage}")
        loader_args, arrays = load_activity_arrays(args, stage)
        videos, velocities, rotational_velocities, _, _ = arrays
        height = loader_args.frame_dim[1] // loader_args.frame_subsampling
        width = loader_args.frame_dim[0] // loader_args.frame_subsampling
        obs_shape = (1, height, width)
        action_dim = velocities.shape[-1] + rotational_velocities.shape[-1]
        world = build_world(base_config, obs_shape, action_dim)
        path = checkpoint_path(args, stage)
        print(f"  Loading {path}")
        load_checkpoint(world, path, device)
        extracted = extract_activity(
            world,
            arrays,
            obs_shape,
            device,
            args.analysis_batch_size,
            args.deterministic_latents,
            args.cell_activity_source,
        )
        result_dir = run_spatial_analysis(
            args, loader_args, stage, *extracted
        )
        print(f"[+] Saved analysis to {result_dir}")
        del world
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(
        "\n[+] Done. Display the summaries with:\n"
        f"    {os.path.basename(os.sys.executable)} show_model_results.py "
        f"{args.model_name}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse staged Lightning world-model checkpoints."
    )
    parser.add_argument("--model-name", default="0824base")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="0", help="CUDA index or 'cpu'")
    parser.add_argument(
        "--stages",
        type=parse_stages,
        default=list(STAGE_DIRS),
        help="comma-separated subset of crawl,walk,run,adult",
    )
    parser.add_argument("--load-last", action="store_true")
    parser.add_argument("--analysis-batch-size", type=int, default=8)
    parser.add_argument(
        "--deterministic-latents",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--activity-transform",
        choices=("identity", "softplus", "minmax", "halfshift"),
        default="minmax",
    )
    parser.add_argument(
        "--cell-activity-source",
        choices=tuple(CELL_ACTIVITY_DESCRIPTIONS),
        default="deterministic",
        help=(
            "CRSSMV4 states used for place/HD/conjunctive-cell analysis. "
            "lower/upper correspond to precise/coarse; stochastic means the "
            "posterior stochastic state."
        ),
    )
    parser.add_argument("--ratemap-norm", choices=("minmax", "sum"), default="minmax")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--env", default=None)
    parser.add_argument("--env-dim", type=float, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--bptt-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parsed = parser.parse_args()
    if parsed.analysis_batch_size < 1:
        parser.error("--analysis-batch-size must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
