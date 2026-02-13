import numpy as np
import os
from ratinabox.Agent import Agent
from ratinabox.Neurons import *
from ratinabox.contribs.FieldOfViewNeurons import *
import pathlib

def calculate_thetas_smooth(pos, win_size):
    pad = int(np.floor(win_size/2))

    thetas = []

    for idx in range(pad):
        a = pos[idx]
        b = pos[idx+1]

        dy = b[1] - a[1]
        dx = b[0] - a[0]
        if dx == 0:
            theta = 0 if dy > 0 else np.pi
        else:
            theta = np.arctan(dy/dx)

        if dx > 0 : theta -= np.pi/2
        elif dx < 0 : theta += np.pi/2

        thetas.append(theta)

    for p in np.lib.stride_tricks.sliding_window_view(pos, win_size, axis=0):
        dy = np.mean(np.diff(p[1, :]))
        dx = np.mean(np.diff(p[0, :]))
        
        if dx == 0:
            theta = 0 if dy > 0 else np.pi
        else:
            theta = np.arctan(dy/dx)

        if dx > 0 : theta -= np.pi/2
        elif dx < 0 : theta += np.pi/2

        thetas.append(theta)

    for idx in range(len(pos)-pad, len(pos)-1):
        a = pos[idx]
        b = pos[idx+1]

        dy = b[1] - a[1]
        dx = b[0] - a[0]
        if dx == 0:
            theta = 0 if dy > 0 else np.pi
        else:
            theta = np.arctan(dy/dx)

        if dx > 0 : theta -= np.pi/2
        elif dx < 0 : theta += np.pi/2

        thetas.append(theta)

    thetas.append(thetas[-1])

    thetas = np.array(thetas)
    
    return thetas

def calculate_thetas(pos):
    thetas = []

    for idx in range(len(pos)-1):
        a = pos[idx]
        b = pos[idx+1]

        dy = b[1] - a[1]
        dx = b[0] - a[0]
        theta = np.arctan(dy/dx)

        if dx > 0:
            theta -= np.pi/2
        else:
            theta += np.pi/2

        thetas.append(theta)

    thetas.append(thetas[-1])

    thetas = np.array(thetas)
    
    return thetas

def calculate_rot_velocity(thetas):
    rot_velocity = np.zeros_like(thetas, dtype=np.float32)
    for idx in range(1, len(thetas)):
        diff = thetas[idx] - thetas[idx-1]
        if diff > np.pi:
            diff -= 2*np.pi
        elif diff < -np.pi:
            diff += 2*np.pi
        rot_velocity[idx] = diff
    return rot_velocity

def run_simulation(
    behaviour_scheme,
    env_name, env_dim, env_eps,
    env,
    agent_params,
    seconds,
    fps,
    exp_dir,
    seed,
    smooth_theta,
    save_experiment,
    increase_fps=None
):
    np.random.seed(seed)

    if increase_fps is not None:
        if increase_fps < fps or increase_fps%fps != 0:
            raise Exception("Increase FPS must be greater than original FPS and a divisor of it")

    if save_experiment:
        print("[*] Saving experiment")
        pathlib.Path(exp_dir).mkdir(parents=True, exist_ok=True)

    # env.plot_environment()

    agent = Agent(env, params=agent_params)
    agent.velocity /= 1e3 # slow down the agent when it first starts

    t_max = 1*seconds # seconds
    if increase_fps is not None:
        print(f"[!] Increasing FPS to {increase_fps}")
        subsamp = int(increase_fps/fps)
        fps = increase_fps
    dt = 1./fps
    
    for i in range(int(t_max/dt)):
        if behaviour_scheme != 'gridsearch':
            if i%int(t_max/dt/5) == 0 : print(f"{i/(t_max/dt)*100:.1f} %")
        agent.update(dt=dt)

    timestamps = np.array(agent.history['t'])
    positions = np.array(agent.history['pos'])
    if np.isnan(positions).any():
        raise Exception("NaN in positions")
    
    if env_name == 'circle':
        print('\tMoving positions to the center of the circle')
        positions += (env_dim/2)
    
    velocities = np.array(agent.history['vel'])

    if smooth_theta > 0:
        smooth_theta = int(smooth_theta*fps) # smooth_theta given in seconds
        thetas = calculate_thetas_smooth(
            positions,
            smooth_theta if smooth_theta%2!=0 else smooth_theta+1
        )
    else:
        thetas = calculate_thetas(positions)

    rot_velocities = calculate_rot_velocity(thetas)
    if behaviour_scheme == 'gridsearch':
        rot_velocities *= fps

    if increase_fps is not None:
        print("\n[!] Switching back to original FPS")
        timestamps = timestamps[subsamp-1::subsamp]
        
        positions = positions[subsamp-1::subsamp]
        thetas = thetas[subsamp-1::subsamp]
        velocities = np.add.reduceat(velocities, np.arange(0, len(velocities), subsamp))/subsamp
        velocities = np.vstack([velocities[1:], velocities[-1]])
        rot_velocities = np.add.reduceat(rot_velocities, np.arange(0, len(rot_velocities), subsamp))
        rot_velocities = np.hstack([rot_velocities[1:],rot_velocities[-1]])
        rot_velocities[rot_velocities > np.pi] -= 2*np.pi
        rot_velocities[rot_velocities < -np.pi] += 2*np.pi

    if env_name == 'box':
        positions += env_eps

    if save_experiment:
        np.save(os.path.join(exp_dir, "timestamps.npy"), timestamps)
        np.save(os.path.join(exp_dir, "positions.npy"), positions)
        np.save(os.path.join(exp_dir, "thetas.npy"), thetas)
        np.save(os.path.join(exp_dir, "velocities.npy"), velocities)
        np.save(os.path.join(exp_dir, "rot_velocities.npy"), rot_velocities)
    
    return agent, positions, velocities, rot_velocities, thetas
