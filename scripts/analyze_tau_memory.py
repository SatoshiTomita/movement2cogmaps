"""Relate CTRNN memory timescale (tau) to cognitive-map metrics.

Uses the saved epoch-1500 activity analyses of the CTRNN tau sweep
(500 units, sigmoid, seed s01; configs differ only in ``ctrnn_tau``).
No model is trained or re-run.

Per (tau, stage) it reports
  * effective memory:
      - unit autocorrelation timescale (lag where the ACF drops below 1/e)
      - past-displacement decoding: R^2 of decoding p_(t-k) - p_t from h_t,
        after regressing out what the current state (p_t, heading, velocity)
        already explains, cross-validated across trials
  * cognitive-map metrics: sRSA, position / HD decoding error, cell counts,
    median SIr / SId / RVL, final test loss
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

STAGES = ("crawl", "walk", "run", "adult")
HISTORY = {"crawl": "vanilla", "walk": "crawl", "run": "crawl_walk",
           "adult": "crawl_walk_run"}
TAUS = (1, 2, 3, 4, 8, 16, 32, 64)
# k=1 is excluded: p_(t-1) - p_t equals -velocity, which is regressed out.
LAGS = (2, 3, 5, 8, 12, 20, 30)
ACF_MAX_LAG = 150


def find_runs(data_root, stage, tau):
    """Return {prefix: activity_dir} for one stage/tau (prefix like '1005')."""
    base = data_root / "box" / stage / "predictions" / "box_messy" / HISTORY[stage]
    runs = {}
    for name in sorted(os.listdir(base)):
        if "CTRNN_100" not in name or not name.endswith(f"_s01_tau{tau}"):
            continue
        act = base / name / f"act_{stage}_epoch1500"
        if act.is_dir():
            runs[name.split("CTRNN_")[1][:4]] = act
    return runs


def acf_timescale(h):
    """Median over active units of the 1/e crossing lag of the within-trial ACF."""
    h = h - h.mean(axis=1, keepdims=True)
    var = (h ** 2).mean(axis=(0, 1))
    active = var > 1e-8
    h, var = h[..., active], var[active]
    n_t = h.shape[1]
    # FFT autocorrelation per trial and unit, averaged over trials.
    f = np.fft.rfft(h, n=2 * n_t, axis=1)
    ac = np.fft.irfft(f * np.conj(f), axis=1)[:, :ACF_MAX_LAG + 1]
    ac = ac / (n_t - np.arange(ACF_MAX_LAG + 1))[None, :, None]
    acf = ac.mean(axis=0) / var[None, :]
    below = acf < np.exp(-1)
    first = np.where(below.any(axis=0), below.argmax(axis=0), ACF_MAX_LAG + 1)
    return float(np.median(first)), float(active.mean()), acf.mean(axis=1)


def ridge_cv_r2(x, y, groups, alpha=1.0, n_folds=5):
    """Cross-validated R^2 (pooled over outputs) with folds split by trial."""
    folds = np.array_split(np.random.default_rng(0).permutation(np.unique(groups)), n_folds)
    ss_res = ss_tot = 0.0
    for test_trials in folds:
        test = np.isin(groups, test_trials)
        xm, ym = x[~test].mean(0), y[~test].mean(0)
        xs = x[~test].std(0) + 1e-6
        xtr, xte = (x[~test] - xm) / xs, (x[test] - xm) / xs
        w = np.linalg.solve(xtr.T @ xtr + alpha * len(xtr) * np.eye(x.shape[1]),
                            xtr.T @ (y[~test] - ym))
        ss_res += ((y[test] - ym - xte @ w) ** 2).sum()
        ss_tot += ((y[test] - y[test].mean(0)) ** 2).sum()
    return 1.0 - ss_res / ss_tot


def past_decoding(h, pos, theta, alpha):
    """R^2 of decoding the residual past displacement from h_t for each lag."""
    n_trials, n_t, _ = h.shape
    t0 = max(LAGS) + 1
    t = np.arange(t0, n_t)
    trial = np.repeat(np.arange(n_trials), len(t))
    ht = h[:, t].reshape(-1, h.shape[-1])
    vel = pos[:, t] - pos[:, t - 1]
    th = theta[:, t, 0]
    current = np.concatenate([pos[:, t], vel, np.cos(th)[..., None],
                              np.sin(th)[..., None]], axis=-1).reshape(-1, 6)
    # Quadratic expansion so smooth nonlinear dependence on the present is removed too.
    current = np.concatenate([current, (current[:, :, None] * current[:, None, :])
                              .reshape(len(current), -1)], axis=1)
    c1 = np.column_stack([np.ones(len(current)), current])
    out = {}
    for k in LAGS:
        disp = (pos[:, t - k] - pos[:, t]).reshape(-1, 2)
        resid = disp - c1 @ np.linalg.lstsq(c1, disp, rcond=None)[0]
        out[k] = float(ridge_cv_r2(ht, resid, trial, alpha=alpha))
    return out


def map_metrics(act):
    run = act.parent
    d = {"loss_test": float(np.load(run / "loss_test.npy")[-1]),
         "sRSA": float(np.nanmean(np.load(act / "sRSA_values.npy"))),
         "pos_dec_err": float(np.nanmean(np.load(act / "pos_dec_err.npy"))),
         "hd_dec_err": float(np.nanmean(np.load(act / "thet_dec_err.npy"))),
         "n_place": int(len(np.load(act / "indices_place_cells.npy"))),
         "n_hd": int(len(np.load(act / "indices_hd_cells.npy"))),
         "n_conj": int(len(np.load(act / "indices_conjunctive_cells.npy")))}
    for key, f in (("SIr", "place/si.npy"), ("SId", "hd/si.npy"), ("RVL", "hd/rvl.npy")):
        d[key] = float(np.nanmedian(np.load(act / f)))
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("R_stats/tau_memory_results"))
    ap.add_argument("--alpha", type=float, default=1e-2)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows, acf_curves = [], {}
    for stage in STAGES:
        for tau in TAUS:
            for prefix, act in find_runs(args.data_root, stage, tau).items():
                h = np.load(act / "recurrent_activity.npy")
                pos = np.load(act / "positions.npy")
                theta = np.load(act / "thetas.npy")
                ts, frac_active, acf = acf_timescale(h)
                r2 = past_decoding(h, pos, theta, args.alpha)
                row = {"stage": stage, "tau": tau, "run": prefix,
                       "acf_timescale": ts, "frac_active": frac_active,
                       **{f"past_r2_k{k}": v for k, v in r2.items()},
                       "past_r2_mean": float(np.mean(list(r2.values()))),
                       **map_metrics(act)}
                rows.append(row)
                acf_curves[f"{stage}_tau{tau}_{prefix}"] = acf.tolist()
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in row.items()}), flush=True)

    keys = list(rows[0])
    with open(args.out / "tau_memory_metrics.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    (args.out / "acf_curves.json").write_text(json.dumps(acf_curves))

    # Primary series: one run per tau (1006 for tau=1, the only one with all stages).
    primary = [r for r in rows if not (r["tau"] == 1 and r["run"] != "1006")]
    metrics = ["acf_timescale", "past_r2_mean", "past_r2_k5", "past_r2_k12",
               "loss_test", "sRSA", "pos_dec_err", "hd_dec_err", "n_place",
               "SIr", "SId", "RVL"]
    lines = ["stage,metric,spearman_rho_vs_tau,p,n"]
    for stage in STAGES:
        sub = sorted((r for r in primary if r["stage"] == stage), key=lambda r: r["tau"])
        for m in metrics:
            rho, p = stats.spearmanr([r["tau"] for r in sub], [r[m] for r in sub])
            lines.append(f"{stage},{m},{rho:.3f},{p:.4f},{len(sub)}")
        for m in ["sRSA", "pos_dec_err", "SIr", "n_place"]:
            rho, p = stats.spearmanr([r["past_r2_mean"] for r in sub], [r[m] for r in sub])
            lines.append(f"{stage},{m}_vs_past_r2_mean,{rho:.3f},{p:.4f},{len(sub)}")
    (args.out / "tau_spearman.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, m in zip(axes.flat, ["acf_timescale", "past_r2_mean", "loss_test", "sRSA",
                                 "pos_dec_err", "n_place", "SIr", "RVL"]):
        for stage in STAGES:
            sub = sorted((r for r in primary if r["stage"] == stage), key=lambda r: r["tau"])
            ax.plot([r["tau"] for r in sub], [r[m] for r in sub], "o-", label=stage)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("tau (steps)")
        ax.set_title(m)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(args.out / "tau_metrics.png", dpi=120)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    for ax, stage in zip(axes, STAGES):
        for r in sorted((r for r in primary if r["stage"] == stage), key=lambda r: r["tau"]):
            ax.plot(LAGS, [r[f"past_r2_k{k}"] for k in LAGS], "o-", label=f"tau{r['tau']}")
        ax.set_title(f"{stage}: residual past-displacement R^2")
        ax.set_xlabel("lag k (steps)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "past_decoding.png", dpi=120)


if __name__ == "__main__":
    main()
