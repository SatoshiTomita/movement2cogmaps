"""Fit descriptive CTRNN tau-response curves to saved tau-sweep metrics.

The input contains one independent training realization per tau.  Therefore
the fits describe this sweep only; they are not seed-level statistical tests.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STAGES = ("crawl", "walk", "run", "adult")
METRICS = (
    "acf_timescale", "past_r2_mean", "past_r2_k5", "past_r2_k12",
    "loss_test", "sRSA", "pos_dec_err", "hd_dec_err", "n_place", "n_hd",
    "n_conj", "SIr", "SId", "RVL",
)
COUNT_METRICS = {"n_place", "n_hd", "n_conj"}
MODELS = (("constant", 0), ("linear", 1), ("quadratic", 2))
COLORS = {"crawl": "#377eb8", "walk": "#4daf4a", "run": "#ff7f00", "adult": "#984ea3"}


def read_primary_rows(path):
    """Read one deterministic evaluation per stage/tau from the sweep CSV."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    primary = []
    for stage in STAGES:
        for tau in sorted({int(row["tau"]) for row in rows if row["stage"] == stage}):
            candidates = [row for row in rows if row["stage"] == stage and int(row["tau"]) == tau]
            # tau=1 was evaluated twice.  1006 is the common run across all stages.
            chosen = next((row for row in candidates if row["run"] == "1006"), candidates[0])
            primary.append({key: float(value) if key not in {"stage", "run"} else value
                            for key, value in chosen.items()})
    return primary


def fit_polynomial(x, y, degree):
    coefficients = np.polyfit(x, y, degree)
    prediction = np.polyval(coefficients, x)
    rss = float(np.sum((y - prediction) ** 2))
    n = len(y)
    # Mean-function coefficients plus one residual-variance parameter.
    parameter_count = degree + 2
    if rss <= 0 or n <= parameter_count + 1:
        aicc = math.inf
    else:
        aic = n * math.log(rss / n) + 2 * parameter_count
        aicc = aic + 2 * parameter_count * (parameter_count + 1) / (n - parameter_count - 1)
    return coefficients, prediction, rss, aicc


def leave_one_tau_out_rmse(x, y, degree):
    errors = []
    for held_out in range(len(x)):
        keep = np.arange(len(x)) != held_out
        coefficients = np.polyfit(x[keep], y[keep], degree)
        errors.append(float(y[held_out] - np.polyval(coefficients, x[held_out])))
    return float(np.sqrt(np.mean(np.square(errors)))), float(np.max(np.abs(errors)))


def model_rows(x, y):
    fits = []
    for name, degree in MODELS:
        coefficients, prediction, rss, aicc = fit_polynomial(x, y, degree)
        loocv_rmse, loocv_max_error = leave_one_tau_out_rmse(x, y, degree)
        tss = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - rss / tss if tss > 0 else float("nan")
        fits.append({"model": name, "degree": degree, "coefficients": coefficients,
                     "prediction": prediction, "rss": rss, "aicc": aicc,
                     "loocv_rmse": loocv_rmse, "loocv_max_error": loocv_max_error,
                     "r_squared": r_squared})
    minimum_aicc = min(fit["aicc"] for fit in fits)
    # A Delta-AICc <= 2 is treated as practically indistinguishable; select the
    # simpler curve rather than claiming structure that 8 points cannot support.
    eligible = [fit for fit in fits if fit["aicc"] - minimum_aicc <= 2]
    chosen = min(eligible, key=lambda fit: fit["degree"])
    for fit in fits:
        fit["delta_aicc"] = fit["aicc"] - minimum_aicc
        fit["selected"] = fit is chosen
    return fits, chosen


def turning_point(coefficients, degree):
    if degree != 2 or abs(coefficients[0]) < 1e-12:
        return "", ""
    x = -coefficients[1] / (2 * coefficients[0])
    tau = 2 ** x
    if not 1 <= tau <= 64:
        return "", ""
    return tau, "minimum" if coefficients[0] > 0 else "maximum"


def fit_response(metric, values):
    """Transform non-negative count metrics before Gaussian curve fitting."""
    if metric in COUNT_METRICS:
        return np.log1p(values), "log1p"
    return values, "identity"


def inverse_response(response, scale):
    if scale == "log1p":
        return np.maximum(np.expm1(response), 0.0)
    return response


def save_plots(results, output_dir):
    metrics = list(METRICS)
    fig, axes = plt.subplots(3, 5, figsize=(20, 11))
    for ax, metric in zip(axes.flat, metrics):
        for stage in STAGES:
            result = results[(stage, metric)]
            tau, values = result["tau"], result["values"]
            curve_x = np.linspace(0, 6, 241)
            curve_tau = 2 ** curve_x
            curve_y = inverse_response(np.polyval(result["chosen"]["coefficients"], curve_x),
                                       result["fit_scale"])
            ax.scatter(tau, values, color=COLORS[stage], s=24, zorder=3)
            ax.plot(curve_tau, curve_y, color=COLORS[stage], linewidth=1.5,
                    label=f"{stage}: {result['chosen']['model']}")
        ax.set_xscale("log", base=2)
        ax.set_xticks((1, 4, 16, 64))
        ax.set_title(metric)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    fig.supxlabel("CTRNN tau (steps; log2 scale)")
    fig.suptitle("Descriptive fits: simplest model within Delta-AICc <= 2", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "tau_curve_fits.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 5, figsize=(20, 11))
    for ax, metric in zip(axes.flat, metrics):
        for stage in STAGES:
            result = results[(stage, metric)]
            residuals = result["values"] - result["prediction_raw"]
            ax.plot(result["tau"], residuals, "o-", color=COLORS[stage], label=stage)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks((1, 4, 16, 64))
        ax.set_title(metric)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    fig.supxlabel("CTRNN tau (steps; log2 scale)")
    fig.suptitle("Residuals of selected descriptive curves", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "tau_curve_residuals.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("R_stats/tau_memory_results/tau_memory_metrics.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("R_stats/tau_memory_results/curve_fits"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = read_primary_rows(args.input)
    results, candidate_rows, prediction_rows, summary_rows = {}, [], [], []
    for stage in STAGES:
        subset = sorted((row for row in rows if row["stage"] == stage), key=lambda row: row["tau"])
        tau = np.array([row["tau"] for row in subset], dtype=float)
        x = np.log2(tau)
        if len(tau) != 8 or len(set(tau)) != len(tau):
            raise ValueError(f"Expected eight unique tau values for {stage}, found {tau.tolist()}")
        for metric in METRICS:
            values = np.array([row[metric] for row in subset], dtype=float)
            values_for_fit, scale = fit_response(metric, values)
            fits, chosen = model_rows(x, values_for_fit)
            prediction_raw = inverse_response(chosen["prediction"], scale)
            results[(stage, metric)] = {"tau": tau, "values": values, "chosen": chosen,
                                        "fit_scale": scale, "prediction_raw": prediction_raw}
            for fit in fits:
                candidate_rows.append({
                    "stage": stage, "metric": metric, "model": fit["model"], "degree": fit["degree"],
                    "fit_response": scale, "selected": fit["selected"], "aicc": fit["aicc"], "delta_aicc": fit["delta_aicc"],
                    "loocv_rmse": fit["loocv_rmse"], "loocv_max_error": fit["loocv_max_error"],
                    "r_squared": fit["r_squared"], "rss": fit["rss"],
                    "coefficients_descending": ";".join(f"{c:.12g}" for c in fit["coefficients"]),
                })
            vertex_tau, vertex_kind = turning_point(chosen["coefficients"], chosen["degree"])
            summary_rows.append({
                "stage": stage, "metric": metric, "n_unique_tau": len(tau),
                "fit_response": scale, "selected_model": chosen["model"], "aicc": chosen["aicc"],
                "loocv_rmse": chosen["loocv_rmse"], "loocv_max_error": chosen["loocv_max_error"],
                "r_squared": chosen["r_squared"], "turning_tau_if_in_range": vertex_tau,
                "turning_kind": vertex_kind,
            })
            for row, prediction in zip(subset, prediction_raw):
                prediction_rows.append({
                    "stage": stage, "metric": metric, "tau": int(row["tau"]), "run": row["run"],
                    "value": row[metric], "selected_model": chosen["model"],
                    "prediction": prediction, "residual": row[metric] - prediction,
                })

    def write_csv(filename, fieldnames, data):
        with (args.out / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    write_csv("tau_curve_candidates.csv", list(candidate_rows[0]), candidate_rows)
    write_csv("tau_curve_summary.csv", list(summary_rows[0]), summary_rows)
    write_csv("tau_curve_predictions.csv", list(prediction_rows[0]), prediction_rows)
    (args.out / "README.txt").write_text(
        "Each stage/tau value is one trained CTRNN realization, so these are descriptive fits only.\n"
        "x=log2(tau). Candidates are constant, linear, and quadratic least-squares curves.\n"
        "Cell-count metrics are fitted on log1p(count) and shown after inverse transformation.\n"
        "AICc uses mean-curve coefficients plus residual variance; within Delta-AICc <= 2, the simpler model is selected.\n"
        "LOOCV leaves out one tau, not one independent training seed. Do not interpret it as seed-level generalization.\n"
        "The tau=1 duplicate was deterministically identical; run 1006 was retained and 1005 excluded.\n"
    )
    save_plots(results, args.out)


if __name__ == "__main__":
    main()
