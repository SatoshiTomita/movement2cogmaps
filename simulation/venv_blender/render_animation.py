import platform
if 'mac' in platform.platform():
    BASE_DIR = "/Users/USER/vrtopc/"
    DATA_DIR = "/media/data/vrtopc"
else:
    BASE_DIR = "/home/USER/vr_to_pc/"
    DATA_DIR = "/media/data/vrtopc"

import sys
sys.path.append(BASE_DIR)

import os
import argparse
import yaml
from itertools import product
import subprocess

SEEDS = list(range(1, 26 +1))

def main(args):
    BEHAVIOURS = args.behaviours
    
    ENVIRONMENT = args.env

    with open(os.path.join(BASE_DIR, "simulation/config.yaml")) as f:
        config = yaml.safe_load(f)

    FPS = config['FPS']
    ENV_DIM = config['ENV_DIM']
    SECONDS = config['SECONDS']

    ENVIRONMENT_TYPE = ENVIRONMENT.split('_')[0]

    EXPERIMENT_DIRS = [
        os.path.join(DATA_DIR, ENVIRONMENT_TYPE, b,
        f"exp_dim{ENV_DIM}_fps{FPS}_s{SECONDS}_seed{s:02d}") \
        for b, s in product(BEHAVIOURS, SEEDS)
    ]

    N_FRAMES = FPS*SECONDS
    digits = len(str(N_FRAMES))
    pad = ''.join(['#']*digits)

    print(f"The following experiments will be rendered:")
    print("\n".join(EXPERIMENT_DIRS))
    print()
    print("Starting to render now")
    print("Failed to open dir errors are expected, ignore them")
    print()

    for idx, exp_dir in enumerate(EXPERIMENT_DIRS):
        print("Rendering", exp_dir)
        START_FRAME = 1
        END_FRAME = N_FRAMES
        cmd =\
            f"/home/USER/blender-3.6.19-linux-x64/blender --background {ENVIRONMENT}.blend"+\
            f" --python blender_script.py"+\
            f" --frame-start {START_FRAME} --frame-end {END_FRAME}"+\
            f" --render-output \"{os.path.join(exp_dir, f'{ENVIRONMENT}/frame{pad}.png')}\""+\
            f" --render-anim"+\
            f" --log-level 0"+\
            f" -- \"{exp_dir}\" > logs_render.txt;"
        subprocess.run(cmd, shell=True, check=False)
        # print(cmd)

def list_of_strings(arg):
    return arg.strip().split(',')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render animation.')
    parser.add_argument(
        '--behaviours', type=list_of_strings, required=True,
        help='List of behaviours separated by comma.')
    parser.add_argument('--env', type=str, default='box_messy', help='Environment.')
    args = parser.parse_args()

    main(args)
