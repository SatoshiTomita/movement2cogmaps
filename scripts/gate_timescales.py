"""Gate-based retention of trained GRU / LSTM models, expressed as a CTRNN tau.

Runs the saved epoch-1500 checkpoint of every curriculum stage over that
stage's activity dataset (the same data and BPTT state carry-over used by
``RNN_experiment.py --activity_only``) and records, per unit and time step,
the fraction of the previous state that is kept:

    GRU  : h_t = (1 - z_t) * n_t + z_t * h_(t-1)  -> retention z_t
    LSTM : c_t = f_t * c_(t-1) + i_t * g_t         -> retention f_t

The Euler CTRNN keeps ``1 - 1/tau`` of its membrane potential per step, so a
retention ``a`` corresponds to ``tau = 1 / (1 - a)``.

Usage (same model flags as the original training command), e.g.:
    CUDA_VISIBLE_DEVICES= .venv311/bin/python scripts/gate_timescales.py \
        --out R_stats/tau_memory_results/gates_GRU267.json -- \
        --architecture gru --latent_dim 267 --name_prefix 0720_267dim \
        --curriculum crawl,walk,run,adult --env box_messy \
        --activity_only --activity_transform minmax
Add ``--model-dir 'data/box/{stage}/predictions/box_messy/*/NAME_{stage}'``
before ``--`` when the run predates the current directory naming.
"""

import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import RNN_experiment  # noqa: E402

RESULTS = {}
OUT = None
MODEL_DIR = None


def _retention_hook(store):
    def hook(module, inputs, outputs):
        x = inputs[0]
        hx = inputs[1] if len(inputs) > 1 else None
        seq = outputs[0]
        is_lstm = isinstance(module, torch.nn.LSTM)
        n_hidden = module.hidden_size
        if hx is None:
            h0 = x.new_zeros(x.shape[0], 1, n_hidden)
        else:
            h0 = (hx[0] if is_lstm else hx).transpose(0, 1)
        h_prev = torch.cat([h0, seq[:, :-1]], dim=1)
        gi = x @ module.weight_ih_l0.T
        gh = h_prev @ module.weight_hh_l0.T
        if module.bias:
            gi = gi + module.bias_ih_l0
            gh = gh + module.bias_hh_l0
        # PyTorch gate order: GRU (r, z, n), LSTM (i, f, g, o).
        idx = 1
        keep = torch.sigmoid(
            gi[..., idx * n_hidden:(idx + 1) * n_hidden]
            + gh[..., idx * n_hidden:(idx + 1) * n_hidden]
        )
        store.append(keep.reshape(-1, n_hidden).double().cpu().numpy())
    return hook


def gate_part(trainer, exp_dir, dataloader_act, bptt_trainer):
    args = trainer.get_args()
    model_dir = exp_dir
    if MODEL_DIR is not None:
        # Older runs use a different directory naming scheme than the
        # current get_model_name(); drop the empty folder it just created.
        model_dir = glob.glob(MODEL_DIR.format(stage=args.behaviour))
        if len(model_dir) != 1:
            raise FileNotFoundError(f"Expected one model dir for {MODEL_DIR}: {model_dir}")
        model_dir = model_dir[0]
        if os.path.isdir(exp_dir) and not os.listdir(exp_dir):
            os.rmdir(exp_dir)
    ckpt = os.path.join(model_dir, "rnn_epoch1500.pth")
    rnn = torch.load(ckpt, weights_only=False, map_location=RNN_experiment.DEVICE)
    for name in ("gc2hidden", "gc2cell"):
        # Checkpoints pickled before grid-cell support lack these attributes.
        if name not in rnn.__dict__ and name not in rnn._modules:
            setattr(rnn, name, None)
    core = getattr(rnn, "gru", None) or getattr(rnn, "lstm")
    store = []
    handle = core.register_forward_hook(_retention_hook(store))
    args.clip_value = None
    bptt_trainer.test_epoch(rnn, dataloader_act, for_trajectory=True)
    handle.remove()

    keep = np.concatenate(store, axis=0)
    unit_keep = keep.mean(axis=0)
    unit_tau = 1.0 / np.clip(1.0 - unit_keep, 1e-6, None)
    step_tau = 1.0 / np.clip(1.0 - keep, 1e-6, None)
    RESULTS[args.behaviour] = {
        "checkpoint": ckpt,
        "gate": "update z" if isinstance(core, torch.nn.GRU) else "forget f",
        "n_samples": int(keep.shape[0]),
        "mean_retention": float(keep.mean()),
        "unit_retention_quantiles": np.quantile(unit_keep, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
        "unit_tau_quantiles": np.quantile(unit_tau, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
        "tau_of_mean_retention": float(1.0 / (1.0 - keep.mean())),
        "median_step_tau": float(np.median(step_tau)),
        "frac_units_tau_gt": {str(t): float((unit_tau > t).mean()) for t in (2, 4, 8, 16, 32)},
        "unit_retention": unit_keep.tolist(),
    }
    print(json.dumps({k: v for k, v in RESULTS[args.behaviour].items() if k != "unit_retention"}), flush=True)
    with open(OUT, "w") as f:
        json.dump(RESULTS, f)
    return exp_dir


if __name__ == "__main__":
    split = sys.argv.index("--")
    OUT = sys.argv[sys.argv.index("--out") + 1]
    if "--model-dir" in sys.argv[:split]:
        MODEL_DIR = sys.argv[sys.argv.index("--model-dir") + 1]
    sys.argv = [sys.argv[0]] + sys.argv[split + 1:]
    # Reuse RNN_experiment's CLI and curriculum loop unchanged, but replace
    # the analysis step that ``main`` calls after building the dataloaders.
    RNN_experiment.activity_part = gate_part
    with open(RNN_experiment.__file__) as f:
        source = f.read()
    main_block = source[source.index("if __name__ == '__main__':"):]
    main_block = main_block.replace("if __name__ == '__main__':", "if True:", 1)
    exec(compile(main_block, RNN_experiment.__file__, "exec"), RNN_experiment.__dict__)
