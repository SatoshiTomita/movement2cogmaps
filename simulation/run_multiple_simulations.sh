#!/bin/bash

cd /home/USER/vr_to_pc/simulation/riab_simulation;
python riabox_simulation.py --behaviour crawl --environment box --save_experiment;
cd /home/USER/vr_to_pc/simulation/venv_blender;
python render_animation.py --behaviours crawl --env box_messy;

cd /home/USER/vr_to_pc/simulation/riab_simulation;
python riabox_simulation.py --behaviour walk --environment box --save_experiment;
cd /home/USER/vr_to_pc/simulation/venv_blender;
python render_animation.py --behaviours walk --env box_messy;

cd /home/USER/vr_to_pc/simulation/riab_simulation;
python riabox_simulation.py --behaviour run --environment box --save_experiment;
cd /home/USER/vr_to_pc/simulation/venv_blender;
python render_animation.py --behaviours run --env box_messy;

cd /home/USER/vr_to_pc/simulation/riab_simulation;
python riabox_simulation.py --behaviour adult --environment box --save_experiment;
cd /home/USER/vr_to_pc/simulation/venv_blender;
python render_animation.py --behaviours adult --env box_messy;
