#!/usr/bin/env python3
"""Export per-state SIr/SId/RVL inputs and optionally run calculate_stats.r.

Metrics are computed independently for each unit, so slicing a combined-state
metric vector is equivalent to selecting those units before metric calculation.
This does not recompute sRSA, decoding, or cell-percentage statistics.
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("walk", "run", "adult")  # Same three groups as figure 4 / calculate_stats.r.
METRICS = {"sir": "place/si.npy", "sid": "hd/si.npy", "rvl": "hd/rvl.npy"}


def state_indices():
    """Actual latent order: lower h, lower z, upper h, upper z."""
    low_h, low_z = np.arange(128), np.arange(128, 192)
    high_h, high_z = np.arange(192, 224), np.arange(224, 240)
    cases = {"GRU_hidden": ("gru", np.arange(267))}
    cases["GRU50_hidden"] = ("gru50", np.arange(50))
    cases["RNN_0716_hidden"] = ("rnn0716", np.arange(500))
    for label, model, h, z in (
        ("RSSM", "rssm", low_h, low_z),
        ("CRSSMV4_all", "crssmv4", np.r_[low_h, high_h], np.r_[low_z, high_z]),
        ("CRSSMV4_precise", "crssmv4", low_h, low_z),
        ("CRSSMV4_coarse", "crssmv4", high_h, high_z),
    ):
        for source, indices in (("combined", np.sort(np.r_[h, z])),
                                ("deterministic", h), ("stochastic", z)):
            cases[f"{label}_{source}"] = (model, indices)
    return cases


def activity_dir(args, model, stage):
    base = args.data_root / "box" / stage / "predictions" / "box_messy"
    if model == "gru50":
        history = {"walk": "crawl", "run": "crawl_walk", "adult": "crawl_walk_run"}
        # These saved analyses evaluate every training stage on crawl inputs.
        return (base / history[stage] / f"GRU_0808_50dim_{stage}"
                / "act_crawl_epoch1500_halfshift")
    if model == "rnn0716":
        name = (f"RNN_0716_{stage}_ft_f1_w9_st10_fss4_do[0,0,0]"
                "_lat500_nlsigmoid_hreg0.0_wreg0.0_s01")
        return base / "vanilla" / name / f"act_{stage}_epoch1500"
    if model == "crssmv4":
        return (base / "world_model_analysis" / f"0824base_{stage}"
                / f"act_{stage}_checkpoint_cells-combined_latents-{args.latents}")
    if model == "rssm" and args.rssm_model:
        canonical = lambda name: re.sub(r"_ft(?=_|$)", "", name)
        candidates = [p for p in base.glob("*/*") if p.is_dir()
                      and canonical(p.name) == canonical(args.rssm_model)]
        if len(candidates) != 1:
            raise ValueError(f"Expected one {args.rssm_model}/{stage}; found {candidates}")
        return candidates[0] / f"act_{stage}_epoch1500_{args.rssm_transform}_cells-combined"
    candidates = [p for p in base.glob("*/*") if p.is_dir() and (
        p.name == f"GRU_0720_267dim_{stage}" if model == "gru"
        else p.name.startswith("RSSM_rssm_baseline_")
        and "_lat128_stoch8_" in p.name and p.name.endswith("_s01_cat8")
    )]
    if len(candidates) != 1:
        raise ValueError(f"Expected one {model}/{stage} model in {base}; found {candidates}")
    suffix = "" if model == "gru" else "_cells-combined"
    return candidates[0] / f"act_{stage}_epoch1500_minmax{suffix}"


def activity_transform(args, model):
    if model == "gru50":
        return "halfshift"
    if model == "rssm" and args.rssm_model:
        return args.rssm_transform
    return "minmax"


def load_inputs(args, models):
    inputs, provenance = {}, {}
    for model, dimension in (("gru", 267), ("gru50", 50), ("rssm", 192), ("crssmv4", 240), ("rnn0716", 500)):
        if model not in models:
            continue
        inputs[model], provenance[model] = {}, {}
        for stage in STAGES:
            directory = activity_dir(args, model, stage)
            # Legacy RNN_0716 outputs predate activity_transform.txt. Preserve
            # their saved metrics without assigning an undocumented transform.
            if model != "rnn0716":
                transform = (directory / "activity_transform.txt").read_text().strip()
                expected_transform = activity_transform(args, model)
                if transform != expected_transform:
                    raise ValueError(f"Expected {expected_transform} activity in {directory}; got {transform}")
            inputs[model][stage] = {}
            for metric, filename in METRICS.items():
                values = np.load(directory / filename, allow_pickle=False)
                if values.shape != (dimension,) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid {metric} in {directory}: expected {dimension} finite values")
                inputs[model][stage][metric] = values
            provenance[model][stage] = str(directory.resolve())
    real = {}
    for filename in ("g_real", "g_real_perc", "sir_real", "sid_real", "rvl_real",
                     "pc_perc_real", "hdc_perc_real", "phdc_perc_real"):
        values = np.load(args.real_data / f"{filename}.npy", allow_pickle=False)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f"Invalid real-data array: {filename}")
        real[filename] = values
    for group_key in ("g_real", "g_real_perc"):
        if not np.array_equal(np.unique(real[group_key]), [1, 2, 3]):
            raise ValueError(f"{group_key} must contain groups 1, 2, 3")
    for name, values in real.items():
        group = "g_real_perc" if "perc" in name else "g_real"
        if values.shape != real[group].shape:
            raise ValueError(f"Group/value length mismatch: {name}")
    return inputs, provenance, real


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--real-data", type=Path, default=ROOT / "R_stats/data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "R_stats/selected_models")
    parser.add_argument("--latents", choices=("sampled", "argmax"), default="sampled",
                        help="CRSSMV4 posterior realization used by the saved analysis")
    parser.add_argument("--run-r", action="store_true", help="Also create stats_put.csv and HTML per case")
    parser.add_argument("--cases", nargs="+", choices=list(state_indices()),
                        help="Export only these cases (default: GRU267/RSSM/CRSSMV4). GRU50_hidden and RNN_0716_hidden are opt-in; GRU50 uses saved crawl-input evaluations; precise = lower; coarse = upper.")
    parser.add_argument("--rssm-model", help="Exact RSSM directory name (_ft is optional); requires --cases RSSM_...")
    parser.add_argument("--rssm-transform", choices=("minmax", "halfshift"), default="minmax",
                        help="Saved activity transform for --rssm-model (default: minmax)")
    args = parser.parse_args()
    cases = state_indices()
    if args.cases:
        cases = {name: cases[name] for name in args.cases}
    else:
        cases.pop("RNN_0716_hidden")
        cases.pop("GRU50_hidden")
    if "GRU50_hidden" in cases and args.output_root.resolve() == (ROOT / "R_stats/selected_models").resolve():
        parser.error("Use a separate --output-root for GRU50_hidden to preserve the baseline results")
    if "RNN_0716_hidden" in cases and args.output_root.resolve() == (ROOT / "R_stats/selected_models").resolve():
        parser.error("Use a separate --output-root for RNN_0716_hidden to preserve the baseline results")
    if args.rssm_model:
        if not args.cases or any(model != "rssm" for model, _ in cases.values()):
            parser.error("--rssm-model requires selecting only RSSM cases with --cases")
        if args.output_root.resolve() == (ROOT / "R_stats/selected_models").resolve():
            parser.error("Use a separate --output-root for --rssm-model to preserve the baseline results")
    rscript = shutil.which("Rscript")
    if args.run_r and rscript is None:
        parser.error("Rscript is unavailable. Install R, DescTools and reticulate, or omit --run-r to export inputs only.")
    try:
        inputs, provenance, real = load_inputs(args, {model for model, _ in cases.values()})
    except (OSError, ValueError) as exc:
        parser.error(f"{exc}\nRun the matching combined-state analyses first (GRU: hidden state).")

    args.output_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for case, (model, indices) in cases.items():
        directory = (args.output_root / case).resolve()
        data = directory / "data"
        data.mkdir(parents=True, exist_ok=True)
        np.save(data / "g_model.npy", np.repeat([1, 2, 3], len(indices)))
        for metric in METRICS:
            np.save(data / f"{metric}_model.npy", np.concatenate([
                inputs[model][stage][metric][indices] for stage in STAGES
            ]))
        for name, values in real.items():
            np.save(data / f"{name}.npy", values)
        metadata = {
            "case": case, "groups": dict(enumerate(STAGES, 1)),
            "units_per_stage": len(indices), "unit_indices": indices.tolist(),
            "activity_transform": activity_transform(args, model),
            "activity_directories": provenance[model],
            "rssm_model": args.rssm_model if model == "rssm" else None,
            "crssmv4_latents": args.latents if model == "crssmv4" else None,
            "real_data_directory": str(args.real_data.resolve()),
            "scope": "SIr/SId/RVL by unit; original real-data tests retained",
        }
        if model == "gru50":
            metadata.update({
                "model_family": "GRU_0808_50dim", "seed": 1, "epoch": 1500,
                "group_meaning": "Training stage; all stages evaluated on crawl activity inputs",
                "activity_behaviour": "crawl",
            })
        if model == "rnn0716":
            metadata.update({
                "activity_transform": None,
                "activity_transform_note": "Legacy saved metrics; activity transform is not recorded.",
                "model_family": "RNN_0716", "seed": 1, "epoch": 1500,
            })
        (directory / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"{case}: {len(indices)} units/stage -> {directory}", flush=True)
        if args.run_r:
            output = directory / "stats_put.csv"
            env = os.environ.copy()
            env.setdefault("RETICULATE_PYTHON", sys.executable)
            subprocess.run([rscript, str(ROOT / "R_stats/calculate_stats.r"),
                            str(directory), str(output)], env=env, check=True)
            title = case.replace("CRSSMV4_precise", "CRSSMV4 下位層（precise）").replace(
                "CRSSMV4_coarse", "CRSSMV4 上位層（coarse）")
            if args.rssm_model:
                title = f"{args.rssm_model} / {case}"
            if model == "gru50":
                title = "GRU_0808_50dim / 学習段階 walk → run → adult（全段階でcrawl入力を評価、halfshift）"
            subprocess.run([sys.executable, str(ROOT / "R_stats/summarize_stats.py"),
                            str(output), "--groups", *STAGES,
                            "--title", f"{title} ({len(indices)} units/stage)"], check=True)
            with output.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    all_rows.append({"case": case, "units_per_stage": len(indices), **row})
    if args.run_r:
        output = args.output_root / "stats_put.csv"
        with output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Combined CSV: {output}")


if __name__ == "__main__":
    main()
