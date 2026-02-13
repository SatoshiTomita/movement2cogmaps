import platform
if 'mac' in platform.platform():
    BASE_DIR = "/Users/USER/vrtopc/"
    DATA_DIR = "/media/data/vrtopc"
else:
    BASE_DIR = "/home/USER/vr_to_pc/"
    DATA_DIR = "/media/data/vrtopc"

import sys
sys.path.append(BASE_DIR)


COLUMNS_RIAB = ['speed_mean', 'speed_ct', 'rs_std', 'thigmotaxis']
COLUMNS_METRIC = [
    'speed_js', 'rs_js', 'tm_js',
]

import yaml
import os
import numpy as np
with open(os.path.join(BASE_DIR, "real_data_exploration/cluster_locomotion/config.yaml")) as f:
    config = yaml.safe_load(f)

DT = 1./config['FPS']

BOX_LENGTH = config['BOX_DIM'] # m

DF_DAMP = 1./(2*config['FPS']) # DF damp (DF^(DF_DAMP*i))
K = config['FPS']*config['THRESHOLD_SEC'] # seconds
B = config['B']
B_HD = config['B_HD']
HALF_IDX = B**2//2
BOX_LENGTH_LOWER_TH = BOX_LENGTH - BOX_LENGTH/B

import random

from real_data_exploration.cluster_locomotion.grid_search_params import *

import argparse
import time
import pandas as pd

from real_data_exploration.utils.process_data import calculate_hist_occ
from real_data_exploration.utils.transition_matrix_helper import (
    generate_tm_loop, reorder_transition_matrix, fold_transition_matrix, reord_folded_transition_matrix
)


def behavioural_metrics(df_data):

    occ_s_dict = df_data['speed'].map(lambda x: calculate_hist_occ(
        x, config['SPEED_BINS'], config['SPEED_RANGE']
    )).to_dict()
    occ_rs_dict = df_data['rs'].map(lambda x: calculate_hist_occ(
        x, config['RS_BINS'], [-np.pi, np.pi]
    )).to_dict()    

    df_data['tm'] = df_data.apply(lambda row: generate_tm_loop(
        row['x'], row['y'],
        BOX_LENGTH, B,
        config['DF'], DF_DAMP, K, config['ATOL'],
        config['SIGMA_SMOOTHING'], subsample=1, plot=False
    )[0], axis=1)

    df_data['tm_reord'] = df_data['tm'].map(lambda x: reorder_transition_matrix(x, B))

    df_data['tm_reord_fold'] = df_data.apply(lambda row: fold_transition_matrix(
        row['tm'], row['tm_reord'], B, HALF_IDX
    )[0], axis=1)

    tm_fold_dict = df_data['tm_reord_fold'].map(lambda x: reord_folded_transition_matrix(x, B)).to_dict()

    return (
        occ_s_dict, occ_rs_dict, tm_fold_dict
    )


def run_grid_search(
    args, idx_start,
    df_dict, df_indices,
    df_dir, df_name, 
    env,
    occ_s_dict, occ_rs_dict,
    tm_fold_dict
):
    from itertools import product
    from simulation.riab_simulation.utils import run_simulation
    from real_data_exploration.utils.grid_search_helper import parametrize_riab_simulations, compare_parameters
    from real_data_exploration.utils.transition_matrix_helper import generate_transition_matrix

    idx_curr = 0
    # start grid search
    for current_values in product(
        SPEED_MEAN, SPEED_CT, ROT_SPEED_STD, THIGMOTAXIS
    ):
        if idx_curr < idx_start:
            if len(df_dict)==0 : raise ValueError("df_dict should be a previously saved dataframe, but it's None.")
            idx_curr += 1
            continue

        s_mean, s_ct, rs_std, thig = current_values
        current_values = list(current_values)
        print(f"{idx_curr+1} of {args.n_combo}")
        
        stime = time.time()
        agent_params = {
            "dt": DT,
            "speed_mean": s_mean,
            "speed_coherence_time": s_ct,
            "rotational_velocity_std": rs_std,
            "rotational_velocity_coherence_time": 0.075,
            "thigmotaxis": thig,
        }
        x_riab = []
        y_riab = []
        speed_riab = []
        rs_riab = []
        tm = np.zeros((B, B, B, B))
        tm_occ = np.zeros((B, B))

        for s in range(args.n_riab_simulations):
            _, p, v, rs, _ = run_simulation(
                'gridsearch',
                'box', config['BOX_DIM'], config['BOX_EPS'],
                env,
                agent_params,
                seconds=config['SECONDS'],
                fps=config['FPS'],
                exp_dir="",
                seed=s,
                smooth_theta=config['SMOOTH_THETA'],
                save_experiment=False,
            )

            speed = np.linalg.norm(v, axis=-1)
            speed_riab.append(speed)

            rs_riab.append(rs)

            x = p[:,0]
            y = p[:,1]
            x_riab.append(x)
            y_riab.append(y)

            _, tm_norm = generate_transition_matrix(
                x, y, BOX_LENGTH, B,
                config['DF'], DF_DAMP, K,
                sigma=config['SIGMA_SMOOTHING']
            )
            tm_norm_occ = np.sum(tm_norm, axis=(-2,-1))
            tm += tm_norm
            tm_occ += tm_norm_occ

        # finalize TM calculations
        tm = np.divide(
            tm.T, tm_occ.T,
            out=np.zeros_like(tm.T), where=tm_occ.T != 0
        ).T
        tmp = np.sum(tm, axis=(-2,-1))
        assert np.allclose(tmp[tmp>0], 1, atol=config['ATOL'])
        
        # reorder and fold transition matrix
        tm_reord = reorder_transition_matrix(tm, B)
        tm_reord_fold_riab, _, _, _ = fold_transition_matrix(tm, tm_reord, B, HALF_IDX)
        tm_fold_riab = reord_folded_transition_matrix(tm_reord_fold_riab, B)
        etime = time.time()
        print(f"RatInABox simulations took: {etime-stime:.2f} s")

        stime = time.time()
        occ_s_riab, occ_rs_riab =\
            parametrize_riab_simulations(
                speed_riab,
                rs_riab,
                config,
            )

        # calculate behavioural metrisc errors
        (
            js_s_dict, js_rs_dict, js_tm_dict
        ) = compare_parameters(
            occ_s_dict, occ_s_riab,
            occ_rs_dict, occ_rs_riab,
            tm_fold_dict, tm_fold_riab,
        )

        etime = time.time()
        print(f"Metrics calculation took: {etime-stime:.2f} s")
        df_dict = {
            **df_dict,
            **{
                (idx_curr, k) : current_values+[
                    js_s_dict[k], js_rs_dict[k], js_tm_dict[k],
                ] for k in js_s_dict.keys()
            }
        }
        
        idx_curr += 1
        if idx_curr % args.save_every == 0:
            print(f"Saving dataframe...")
            df = pd.DataFrame(df_dict).T
            df.columns = COLUMNS_RIAB+COLUMNS_METRIC
            df.index.names = df_indices
            df.to_pickle(os.path.join(df_dir, df_name))

        print('', flush=True)
    
    print(f"Saving dataframe...")
    df = pd.DataFrame(df_dict).T
    df.columns = COLUMNS_RIAB+COLUMNS_METRIC
    df.index.names = df_indices
    df.to_pickle(os.path.join(df_dir, df_name))


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    print()
    print("Grid search space:")
    print()
    print("SPEED_MEAN:", SPEED_MEAN)
    print("SPEED_CT:", SPEED_CT)
    print("ROT_SPEED_STD:", ROT_SPEED_STD)
    print("THIGMOTAXIS:", THIGMOTAXIS)
    print()
    args.n_combo = len(SPEED_MEAN) * len(SPEED_CT) * len(ROT_SPEED_STD) * len(THIGMOTAXIS)

    print("Total number of combinations:", args.n_combo)
    print()

    print("[*] Initializing, this may take a while...")
    print()

    df_dir = os.path.join(DATA_DIR, 'cluster_locomotion', f'by_{args.by}')
    df_data = pd.read_pickle(os.path.join(df_dir, f'data_{args.seed}.pkl'))

    c_idx_col = f'cluster_idx_{args.clusteralgo}'
    df_data = df_data[df_data[c_idx_col] != -1]
    df_data = df_data.groupby(c_idx_col)[['speed', 'rs', 'x', 'y', 'hd']].agg('sum')

    # Behavioural Metrics Definition
    (
        occ_s_dict, occ_rs_dict, tm_fold_dict
    ) = behavioural_metrics(df_data)

    from ratinabox.Environment import Environment
    env = Environment(
        params={'scale':BOX_LENGTH-2*config['BOX_EPS'], 'aspect':1}
    )

    df_dict = {}
    df_name = f'grid_search_{args.clusteralgo}_{args.seed}.pkl'
    idx_start = 0

    df_indices = ['idx', c_idx_col]
    
    try:
        df = pd.read_pickle(os.path.join(df_dir, df_name))
        idx_start = df.index.get_level_values('idx').to_numpy().max()+1
        df_dict = df.to_dict('split')
        df_dict = {
            index: data
            for index, data in zip(df_dict['index'], df_dict['data'])
        }
        print(f"Previous dataframe found. Starting from idx {idx_start}\n")
    except FileNotFoundError:
        print("Previous dataframe not present, starting from zero.")

    # Grid Search
    run_grid_search(
        args, idx_start,
        df_dict, df_indices,
        df_dir, df_name, 
        env,
        occ_s_dict, occ_rs_dict,
        tm_fold_dict,
    )


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--by', type=str, default='day',
        help="By experiment, by day, or by age (exp, day, or age).")
    argparser.add_argument(
        '--clusteralgo', type=str, default='gm',
        help="Which clustering algorithm (spectral, gm) output should the grid search be performed on. Default to 'spectral'.")
    argparser.add_argument(
        '--seed', type=int, default=7,
        help="Seed. Default to 1.")
    argparser.add_argument(
        '--save_every', type=int, default=20,
        help="How often to save results to pickle. Default to 25.")
    argparser.add_argument(
        '--n_riab_simulations', type=int, default=5,
        help="How many RIAB simulations to run for each set of parameters. Default to 5.")
    
    args = argparser.parse_args()

    main(args)
