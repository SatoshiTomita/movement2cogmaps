import platform
if 'mac' in platform.platform():
    BASE_DIR = "/Users/USER/vrtopc/"
    DATA_DIR = "/media/data/vrtopc"
else:
    BASE_DIR = "/home/USER/vr_to_pc/"
    DATA_DIR = "/media/data/vrtopc"

import sys
sys.path.append(BASE_DIR)

import argparse
import os
import yaml
import numpy as np

from ratinabox.Environment import Environment

from simulation.riab_simulation.utils import run_simulation

SEEDS = range(1, 26 +1)

def main(args):
    BEHAVIOUR = args.behaviour
    ENVIRONMENT = args.environment
    SMOOTH_THETA = args.smooth_theta
    INCREASE_FPS = args.increase_fps

    SAVE_EXPERIMENT = args.save_experiment

    if not SAVE_EXPERIMENT:
        print("\n[!!!] WARNING: not saving the experiment, use --save_experiment flag\n")

    with open(f"{BASE_DIR}/simulation/config.yaml") as f:
        config = yaml.safe_load(f)

    behaviour_groups = config['BEHAVIOURS']

    EXPERIMENT_NAMES = [
        f"exp_dim{config['ENV_DIM']}_fps{config['FPS']}_s{config['SECONDS']}_seed{s:02d}"\
        for s in SEEDS
    ]

    agent_params = {
        "dt": 1./config['FPS'], # (s)
        "speed_mean": behaviour_groups[BEHAVIOUR]['SPEED_MEAN'], # 2D: Scale parameter and mode of the speed Rayleigh distribution (m*s^-1)
        "speed_coherence_time": behaviour_groups[BEHAVIOUR]['SPEED_CT'], # Timescale over which speed (1D or 2D) decoheres under random motion (s)
        "rotational_velocity_std": behaviour_groups[BEHAVIOUR]['ROT_SPEED_STD']*np.pi, # std of rotational velocity Normal distribution (rad s^−1)
        "rotational_velocity_coherence_time": behaviour_groups[BEHAVIOUR]['ROT_SPEED_CT'], # Timescale over which rotational velocities
                                                    # decoheres under random motion (s)
        "thigmotaxis": behaviour_groups[BEHAVIOUR]['THIGMOTAXIS'],
        "wall_repel_distance": 0.1,
    }

    EXP_FOLDERS = [
        os.path.join(DATA_DIR, ENVIRONMENT, BEHAVIOUR, exp_name, "riab_simulation") \
        for exp_name in EXPERIMENT_NAMES
    ]

    print("Experiment folders:")
    print(EXP_FOLDERS)

    if ENVIRONMENT == 'box':
        env = Environment(params={'scale':config['ENV_DIM']-2*config['ENV_EPS'], 'aspect':1})

    agent, positions, velocities, rot_velocities, thetas, place_cells, hd_cells, whiskers =\
        [], [], [], [], [], [], [], []
    idx = len(SEEDS)-1

    for s, exp_folder in zip(SEEDS, EXP_FOLDERS):
        print(f"Seed: {s}")
        a, p, v, rv, t = run_simulation(
            BEHAVIOUR,
            ENVIRONMENT, config['ENV_DIM'], config['ENV_EPS'],
            env,
            agent_params,
            config['SECONDS'],
            config['FPS'],
            exp_folder,
            s,
            smooth_theta=SMOOTH_THETA,
            save_experiment=SAVE_EXPERIMENT,
            increase_fps=INCREASE_FPS
        )
        agent.append(a)
        positions.append(p)
        velocities.append(v)
        rot_velocities.append(rv)
        thetas.append(t)
        idx -= 1
        print()

        # Copy the config file to each experiment folder
        if SAVE_EXPERIMENT:
            os.system(f"cp {BASE_DIR}/simulation/config.yaml {'/'.join(exp_folder.split('/')[:-1])}/config.yaml")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--behaviour', type=str, required=True)
    parser.add_argument('--environment', type=str, default="box")
    parser.add_argument('--smooth_theta', type=float, default=1.) # seconds
    parser.add_argument('--increase_fps', type=int, default=50)

    parser.add_argument('--save_experiment', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    main(args)