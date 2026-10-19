import glob
import os
import argparse
import yaml
from itertools import product
from pathlib import Path
import subprocess

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

def main(args):
    BEHAVIOURS = args.behaviours

    ENVIRONMENT = args.env

    with open(REPO_ROOT / "simulation/config.yaml") as f:
        config = yaml.safe_load(f)

    FPS = config['FPS']
    ENV_DIM = config['ENV_DIM']
    SECONDS = config['SECONDS']

    ENVIRONMENT_TYPE = ENVIRONMENT.split('_')[0]

    EXPERIMENT_DIRS = [
        os.path.join(args.data_root, ENVIRONMENT_TYPE, b,
        f"exp_dim{ENV_DIM}_fps{FPS}_s{SECONDS}_seed{s:02d}") \
        for b, s in product(BEHAVIOURS, args.seeds)
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

    missing = [
        d for d in EXPERIMENT_DIRS
        if not all(os.path.exists(os.path.join(d, f)) for f in
                   ("config.yaml", "riab_simulation/positions.npy", "riab_simulation/thetas.npy"))
    ]
    if missing:
        raise FileNotFoundError(
            "Missing simulation inputs (run riabox_simulation.py first):\n" + "\n".join(missing)
        )

    for idx, exp_dir in enumerate(EXPERIMENT_DIRS):
        print("Rendering", exp_dir)
        START_FRAME = args.frame_start
        END_FRAME = args.frame_end or N_FRAMES
        # Blender resolves relative paths against $PWD rather than the subprocess cwd,
        # so pass absolute paths; textures are resolved relative to the .blend file.
        cmd = [
            str(args.blender), "--background", str(SCRIPT_DIR / f"{ENVIRONMENT}.blend"),
            "--python", str(SCRIPT_DIR / "blender_script.py"),
            "--frame-start", str(START_FRAME), "--frame-end", str(END_FRAME),
            "--render-output", os.path.join(exp_dir, f"{ENVIRONMENT}/frame{pad}.png"),
            "--render-anim",
            "--log-level", "0",
            "--", exp_dir,
        ]
        if args.dry_run:
            print(" ".join(cmd))
            continue
        with open(SCRIPT_DIR / "logs_render.txt", "w") as log:
            subprocess.run(cmd, cwd=SCRIPT_DIR, stdout=log, check=False)
        n_rendered = len(glob.glob(os.path.join(exp_dir, ENVIRONMENT, "frame*.png")))
        print(f"  {n_rendered} frames in {os.path.join(exp_dir, ENVIRONMENT)}")

def list_of_strings(arg):
    return arg.strip().split(',')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render animation.')
    parser.add_argument(
        '--behaviours', type=list_of_strings, required=True,
        help='List of behaviours separated by comma.')
    parser.add_argument('--env', type=str, default='box_messy', help='Environment.')
    parser.add_argument(
        '--data-root', type=Path, default=REPO_ROOT / "data",
        help='Directory containing <env type>/<behaviour>/exp_* folders.')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 26 +1)))
    parser.add_argument(
        '--blender', type=Path, default=REPO_ROOT / "tools/blender/blender",
        help='Path to the Blender executable.')
    parser.add_argument('--frame-start', type=int, default=1)
    parser.add_argument(
        '--frame-end', type=int, default=None,
        help='Last frame to render (default: FPS*SECONDS).')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print the Blender commands without running them.')
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()

    main(args)
