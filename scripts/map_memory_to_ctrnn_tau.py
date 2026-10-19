"""Express the activity-based memory of GRU / LSTM models as a CTRNN tau.

The CTRNN tau sweep (500 units, seed s01) is the reference.  For every model
and stage the same two memory measures as ``analyze_tau_memory.py`` are
computed on the saved epoch-1500 activity:
  * ACF timescale: lag where the within-trial unit autocorrelation drops below
    1/e (linearly interpolated between lags), median over active units
  * residual past-displacement decoding R^2 (lags 2-30), from all units and
    from the first ``--n-pcs`` principal components so that models with
    different unit counts are compared at equal dimensionality
Each model value is then mapped onto the CTRNN curve of the same stage.
All models are evaluated on identical trajectories (checked).
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_tau_memory import ACF_MAX_LAG, HISTORY, LAGS, TAUS, find_runs, past_decoding  # noqa: E402

STAGES = ("crawl", "walk", "run", "adult")
MODELS = {
    "RNN500": ("vanilla/RNN_0716_{stage}_ft_f1_w9_st10_fss4_do[[]0,0,0]_lat500_nlsigmoid_hreg0.0_wreg0.0_s01",
               "act_{stage}_epoch1500", "latent_activity.npy"),
    "GRU100": ("*/GRU_0907_100dim_{stage}", "act_{stage}_epoch1500_halfshift", "recurrent_activity.npy"),
    "GRU267": ("*/GRU_0720_267dim_{stage}", "act_{stage}_epoch1500_minmax", "recurrent_activity.npy"),
    "LSTM221": ("*/{stage}_LSTM_LSTM0930", "act_{stage}_epoch1500_halfshift", "recurrent_activity.npy"),
}


def acf_timescale_interp(h):
    """Median over active units of the interpolated 1/e crossing lag."""
    h = h - h.mean(axis=1, keepdims=True)
    var = (h ** 2).mean(axis=(0, 1))
    h, var = h[..., var > 1e-8], var[var > 1e-8]
    n_t = h.shape[1]
    f = np.fft.rfft(h, n=2 * n_t, axis=1)
    ac = np.fft.irfft(f * np.conj(f), axis=1)[:, :ACF_MAX_LAG + 1]
    ac = ac / (n_t - np.arange(ACF_MAX_LAG + 1))[None, :, None]
    acf = ac.mean(axis=0) / var[None, :]
    thr = np.exp(-1)
    out = []
    for u in range(acf.shape[1]):
        below = np.nonzero(acf[:, u] < thr)[0]
        if len(below) == 0:
            out.append(ACF_MAX_LAG + 1)
            continue
        k = below[0]
        a0, a1 = acf[k - 1, u], acf[k, u]
        out.append(k - 1 + (a0 - thr) / (a0 - a1))
    return float(np.median(out))


def pca_project(h, n_pcs):
    x = h.reshape(-1, h.shape[-1]).astype(np.float64)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    _, _, vt = np.linalg.svd(x[::7], full_matrices=False)
    return (x @ vt[:n_pcs].T).reshape(h.shape[0], h.shape[1], -1)


def measures(act, fname, n_pcs, alpha):
    h = np.load(act / fname)
    pos = np.load(act / "positions.npy")
    theta = np.load(act / "thetas.npy")
    full = past_decoding(h, pos, theta, alpha)
    pcs = past_decoding(pca_project(h, n_pcs), pos, theta, alpha)
    return {"n_units": int(h.shape[-1]), "acf": acf_timescale_interp(h),
            **{f"r2_k{k}": v for k, v in full.items()},
            **{f"r2pc_k{k}": v for k, v in pcs.items()},
            "r2_mean": float(np.mean(list(full.values()))),
            "r2pc_mean": float(np.mean(list(pcs.values())))}, pos


def interp_tau(value, taus, values):
    """Invert a monotone-increasing curve on log2(tau); clip outside the range."""
    logt = np.log2(taus)
    order = np.argsort(values)
    v, lt = np.asarray(values)[order], logt[order]
    if value <= v[0]:
        return f"<={taus[order[0]]:g}"
    if value >= v[-1]:
        return f">={taus[order[-1]]:g}"
    return f"{2 ** np.interp(value, v, lt):.2f}"


def nearest_profile(model_row, ref_rows, prefix):
    keys = [f"{prefix}_k{k}" for k in LAGS]
    dist = {r["tau"]: float(np.linalg.norm([model_row[k] - r[k] for k in keys])) for r in ref_rows}
    best = min(dist, key=dist.get)
    return best, dist[best]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("R_stats/tau_memory_results"))
    ap.add_argument("--n-pcs", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=1e-2)
    args = ap.parse_args()

    rows = []
    for stage in STAGES:
        ref_pos = None
        for tau in TAUS:
            runs = find_runs(args.data_root, stage, tau)
            prefix = "1006" if tau == 1 and "1006" in runs else sorted(runs)[0]
            m, pos = measures(runs[prefix], "recurrent_activity.npy", args.n_pcs, args.alpha)
            ref_pos = pos if ref_pos is None else ref_pos
            rows.append({"stage": stage, "model": f"CTRNN_tau{tau}", "tau": tau, **m})
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()
                              if not k.startswith("r2_k") and not k.startswith("r2pc_k")}), flush=True)
        for name, (run_glob, act_name, fname) in MODELS.items():
            base = args.data_root / "box" / stage / "predictions" / "box_messy"
            found = glob.glob(str(base / run_glob.format(stage=stage)))
            if len(found) != 1:
                print(f"skip {name} {stage}: {found}", flush=True)
                continue
            act = Path(found[0]) / act_name.format(stage=stage)
            if not (act / fname).exists():
                print(f"skip {name} {stage}: no {act / fname}", flush=True)
                continue
            m, pos = measures(act, fname, args.n_pcs, args.alpha)
            if not np.allclose(pos, ref_pos):
                raise ValueError(f"{name} {stage}: trajectories differ from the CTRNN reference")
            rows.append({"stage": stage, "model": name, "tau": None, **m})
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()
                              if not k.startswith("r2_k") and not k.startswith("r2pc_k")}), flush=True)

    (args.out / "model_memory_metrics.json").write_text(json.dumps(rows, indent=1))

    lines = ["stage,model,n_units,acf,tau_from_acf,r2pc_mean,tau_from_r2pc_rising,nearest_tau_r2pc_profile,profile_dist"]
    for stage in STAGES:
        ref = [r for r in rows if r["stage"] == stage and r["tau"] is not None]
        taus = np.array([r["tau"] for r in ref], dtype=float)
        rising = [r for r in ref if r["tau"] <= 8]
        for r in rows:
            if r["stage"] != stage:
                continue
            t_acf = interp_tau(r["acf"], taus, [x["acf"] for x in ref])
            t_r2 = interp_tau(r["r2pc_mean"], np.array([x["tau"] for x in rising], dtype=float),
                              [x["r2pc_mean"] for x in rising])
            best, dist = nearest_profile(r, ref, "r2pc")
            lines.append(f"{stage},{r['model']},{r['n_units']},{r['acf']:.2f},{t_acf},"
                         f"{r['r2pc_mean']:.3f},{t_r2},{best},{dist:.3f}")
    (args.out / "model_tau_equivalent.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
