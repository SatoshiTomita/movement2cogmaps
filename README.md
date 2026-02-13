# Movement to Cognitive Maps

Code repository for the paper _From movement to cognitive maps: recurrent neural networks reveal how locomotor development shapes hippocampal spatial coding_, accepted at ICLR 2026 (oral).

Read it at:
- [OpenReview](https://openreview.net/forum?id=8bM7MkxJee) (includes reviews)
- [bioRxiv](https://www.biorxiv.org/content/10.64898/2025.12.30.696864v1)

## Getting started

All notebooks include pre-computed output, so you can browse the results without running anything. To re-run the code, install Conda and create the environment:

```bash
conda env create -f environment.yaml
```

Simulated data is available on [Zenodo](https://zenodo.org/records/15496128). Download it and set the `DATA_DIR` variable in the scripts accordingly (default: `/media/data/vrtopc`). Similarly, adjust `BASE_DIR` to point to your local clone of the repository.

**Note:** The code has been tested on Linux and macOS only.

## Repository structure

```
├── real_data_exploration/
│   └── cluster_locomotion/   # Clustering analysis of locomotor development
│                              # (notebooks 01–04, in execution order;
│                              #  requires experimental data not released
│                              #  with this work; pre-computed outputs included)
│
├── simulation/                # Trajectory simulations of locomotion stages
│                              # run via run_multiple_simulations.sh
│                              # (requires Blender 3.6)
│
├── architectures/             # RNN model definition, training loops, datasets, loss
│
├── RNN_experiment.py          # Train RNNs and analyse hidden-unit activity
├── run_paper_training.sh      # Shell commands to reproduce all paper training runs
├── generate_figure*.ipynb     # Figures 2–5 and supplementary figures
│                              # (Figure 1 is generated during clustering)
│
├── likelihood_test.ipynb      # Likelihood-ratio tests
├── utils/                     # Shared utilities (metrics, plotting, ...)
└── R_stats/                   # Statistical tests run in R
```

## Citation

```bibtex
@inproceedings{abrate2026movement,
    title     = {From movement to cognitive maps: recurrent neural networks reveal how locomotor development shapes hippocampal spatial coding},
    author    = {Abrate, Marco P and Muessig, Laurenz and Bassett, Joshua P and Tan, Hui Min and Cacucci, Francesca and Wills, Thomas J and Barry, Caswell},
    booktitle = {International Conference on Learning Representations},
    year      = {2026},
    url       = {https://openreview.net/forum?id=8bM7MkxJee}
}
```
