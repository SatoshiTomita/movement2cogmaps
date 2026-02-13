import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from utils.metrics import bin_data_size

def plot_trajectory(ax, x, y, ticks, box_length, x_max, y_max, title):
    ax.scatter(x, y, s=1, color='blue')
    ax.set_xlim([0, box_length])
    ax.set_ylim([0, box_length])

    ax.axvline(x_max, 0, 1, ls='--', color='k', alpha=.5)
    ax.axhline(y_max, 0, 1, ls='--', color='k', alpha=.5)

    ax.set_xticks(ticks, ticks)
    ax.set_yticks(ticks, ticks)
    ax.grid()
    ax.set_title(
        title, fontsize='small'
    )

def plot_box_size_hist(box_size, box_length):
    x_min, y_min = 1_000, 1_000
    x_max, y_max = 0, 0
    for k, bs_age in box_size.items():
        if len(bs_age.keys()) <= 1:
            print(f"{k} has no data or 1 datapoint")
            continue
        fig, axs = plt.subplots(1, len(bs_age.keys()), figsize=(20, 5))
        idx = 0
        for age in sorted(bs_age.keys()):
            bs = np.array(bs_age[age])
            axs[idx].boxplot(bs[:, -2:])
            axs[idx].set_title(age)
            axs[idx].set_ylim([0.55, box_length])
            axs[idx].set_yticks(
                np.arange(0.55, box_length, 0.01), np.round(np.arange(0.55, box_length, 0.01), 2)
            )
            idx += 1

            x_min_curr, y_min_curr, x_max_curr, y_max_curr = (
                np.min(bs[:, 0]), np.min(bs[:, 1]), np.max(bs[:, 2]), np.max(bs[:, 3])
            )
            if x_max_curr > x_max : x_max = x_max_curr
            if y_max_curr > y_max : y_max = y_max_curr
            if x_min_curr < x_min : x_min = x_min_curr
            if y_min_curr < y_min : y_min = y_min_curr
        plt.suptitle(k)
        plt.show()
    print()
    print('X (min, max):', x_min, x_max)
    print('Y (min, max):', y_min, y_max)

def filter_trial(
    trial, ld, k, age,
    smooth_theta, fps, duration_range, box_length, box_eps, box_tol, box_length_lower_th
):
    ld_short = ld[:52]
    if (
        (ld == 'SCAN structure - adn_struct' and k == 'r274' and age == 13 and trial['name'] == 1) or
        (ld == 'SCAN structure - adn_struct' and k == 'r322' and age == 18 and trial['name'] == 0) or
        (ld == 'SCAN structure - adn_struct' and k == 'r337' and age == 12 and trial['name'] == 1) or
        (ld == 'SCAN structure - presub_pup_oldrats_struct' and k == 'r1756' and age == 11 and trial['name'] == 0) or
        (ld == 'SCAN structure - presub_pup_oldrats_struct' and k == 'r1762' and age == 11 and trial['name'] == 2) or
        (ld == 'rawData_ento_all_pup_struct' and k == 'r1579' and age == 16 and trial['name'] == 0)
    ):
        print('Skipping outlier', ld_short, k, age, trial['name'])
        return None

    duration = trial['duration']
    if trial['sample_rate'] != fps or trial['environment'] != 'hp' or duration < duration_range[0]:
        print(f"Skipping {ld_short}, {k}, {age}, {trial['name']} because "+\
            f"sample rate is not {fps} Hz, environment is not hp, or duration < {duration_range[0]} s")
        return None

    ppm = trial['ppm']
    x = trial['x']/ppm
    y = trial['y']/ppm
    speed = trial['speed']/100 # speed is already cm/s (NOT pixels/s)
    hd = trial['hd']

    # remove NaNs, this is important for Elly's data, which contains some NaNs
    idx_nan = np.where(np.isnan(x) | np.isnan(y) | np.isnan(speed) | np.isnan(hd))[0]
    if len(idx_nan)/len(x) > 0.1:
        print(f"Skipping {ld_short}, {k}, {age}, {trial['name']} because >10% of NaNs")
        return None
    x = np.delete(x, idx_nan)
    y = np.delete(y, idx_nan)
    speed = np.delete(speed, idx_nan)
    hd = np.delete(hd, idx_nan)
    duration = int(len(x)/fps)


    if duration > duration_range[1]:
        x = x[:duration_range[1]*fps]
        y = y[:duration_range[1]*fps]
        speed = speed[:duration_range[1]*fps]
        hd = hd[:duration_range[1]*fps]
        duration = duration_range[1]

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    count_x_closetomin = np.sum(np.isclose(x, x_min, atol=box_tol))
    count_x_closetomax = np.sum(np.isclose(x, x_max, atol=box_tol))
    count_y_closetomin = np.sum(np.isclose(y, y_min, atol=box_tol))
    count_y_closetomax = np.sum(np.isclose(y, y_max, atol=box_tol))

    # print(f'before conversion: ({x_min:.4f}, {x_max:.4f}), ({y_min:.4f}, {y_max:.4f})')
    # print(f'count_x_closetomin: {count_x_closetomin}, count_x_closetomax: {count_x_closetomax}')
    # print(f'count_y_closetomin: {count_y_closetomin}, count_y_closetomax: {count_y_closetomax}')

    
    if x_min < box_eps and x_min >= 0 and count_x_closetomin >= 3*fps:
        # print('shifting right because x_min is too close to 0')
        x += (box_eps-x_min)
    elif x_min > box_eps and count_x_closetomin >= 3*fps:
        # print('shifting left because x_min is too far from 0')
        x -= (x_min-box_eps)
    elif x_max > (box_length-box_eps) and count_x_closetomax >= 3*fps:
        # print('shifting left because x_max is too close to box_length')
        x -= (x_max-box_length+box_eps)

    if y_min < box_eps and y_min >= 0 and count_y_closetomin >= 3*fps:
        # print('shifting up because y_min is too close to 0')
        y += (box_eps-y_min)
    elif y_min > box_eps and count_y_closetomin >= 3*fps:
        # print('shifting down because y_min is too far from 0')
        y -= (y_min-box_eps)
    elif y_max > (box_length-box_eps) and count_y_closetomax >= 3*fps:
        # print('shifting down because y_max is too close to box_length')
        y -= (y_max-box_length+box_eps)
    
    # rescaling if position is just a little (box_eps) outside the box
    if x_min >= -box_eps and x_min < box_eps: 
        x = (x_max-box_eps)*(x-x_min)/(x_max-x_min) + box_eps
        x_min = np.min(x)
    if x_max > (box_length - box_eps) and x_max <= (box_length + box_eps):
        x = ((box_length-box_eps)-box_eps)*(x-x_min)/(x_max-x_min) + x_min
        x_max = np.max(x)
    if y_min >= -box_eps and y_min < box_eps:
        y = (y_max-box_eps)*(y-y_min)/(y_max-y_min) + box_eps
        y_min = np.min(y)
    if y_max > (box_length - box_eps) and y_max <= (box_length + box_eps):
        y = ((box_length-box_eps)-box_eps)*(y-y_min)/(y_max-y_min) + y_min
        y_max = np.max(y)

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    # print(f'after rescaling: ({x_min:.4f}, {x_max:.4f}), ({y_min:.4f}, {y_max:.4f})')

    if x_max > (box_length-box_eps) or y_max > (box_length-box_eps):
        print(f"Skipping {ld_short}, {k}, {age}, {trial['name']} because "+\
            f"{x_max:.4f} or {y_max:.4f} in a bigger box than {box_length-box_eps}")
        # f, ax = plt.subplots(figsize=(3,3))
        # plot_trajectory(ax, x, y, ticks, np.max(x), np.max(y), f"{k}, {age}, {trial['name']}")
        # plt.show()
        return None
    if x_max < box_length_lower_th or y_max < box_length_lower_th:
        print(f"Skipping {ld_short}, {k}, {age}, {trial['name']} because "+\
            f"{x_max:.4f} or {y_max:.4f} in a smaller box than {box_length_lower_th}")
        # f, ax = plt.subplots(figsize=(3,3))
        # plot_trajectory(ax, x, y, ticks, np.max(x), np.max(y), f"{k}, {age}, {trial['name']}")
        # plt.show()
        return None
    if x_min < box_eps-1e-7 or y_min < box_eps-1e-7:
        print(f"Skipping {ld_short}, {k}, {age}, {trial['name']} because "+\
            f"{x_min:.4f} or {y_min:.4f} smaller than {box_eps}")
        # f, ax = plt.subplots(figsize=(3,3))
        # plot_trajectory(ax, x, y, ticks, 0, 0, f"{k}, {age}, {trial['name']}")
        # plt.show()
        return None

    x = np.array(x)
    y = np.array(y)
    speed = np.array(speed)
    from simulation.riab_simulation.utils import calculate_thetas_smooth
    smooth_theta = int(smooth_theta*fps) # smooth_theta given in seconds
    hd = calculate_thetas_smooth(
        np.stack([x, y], axis=-1),
        smooth_theta if smooth_theta%2!=0 else smooth_theta+1
    )
    from simulation.riab_simulation.utils import calculate_rot_velocity
    rot_speed = calculate_rot_velocity(hd)*fps

    return (x, y, hd, speed, rot_speed, duration)


def exponential(t, tau, K):
    return K * np.exp(-t / tau)
def fit_exponential(ticks, ac):
    tau, _ = curve_fit(exponential, ticks, ac)[0]
    return tau


def moving_average(a, n=3):
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n


def lagged_autocorrelation(x, t_max, fps):
    _min = min(t_max*fps, len(x)-1)
    R = np.zeros(_min, dtype=np.float32)
    R[0] = 1
    for i in range(1, _min):
        R[i] = pearsonr(x[i:], x[:-i])[0]
    return R


def calculate_rot_speed(hd, fps, window):
    rot_speed_est = np.zeros_like(hd)
    rot_speed_est[1:] = np.diff(hd)*fps
    rot_speed_est = moving_average(rot_speed_est, window)
    return rot_speed_est

def calculate_hist_2d_occ(
    data_list_1, data_list_2,
    bin_size, limits, sigma
):
    d = np.stack([
        np.concatenate(data_list_1),
        np.concatenate(data_list_2)
    ], axis=-1)
    occ = bin_data_size(d, bin_size=bin_size, limits=limits)
    occ = gaussian_filter(occ, sigma)
    occ  = np.divide(occ, np.sum(occ), out=np.zeros_like(occ), where=(occ!=0))
    return occ

def calculate_hist_occ(
    data_list,
    data_bins, data_range,
):
    d = np.concatenate(data_list)
    d = d[d!=0]
    occ, _ = np.histogram(d, bins=data_bins, range=data_range, density=True)
    occ = occ.astype(np.float64)

    return occ

def calculate_autocorrelation_tau(
    data_list,
    t_max, fps, data_ac_ticks
):
    ac = [lagged_autocorrelation(
        (d-np.mean(d))/np.std(d),
        t_max=t_max, fps=fps
    ) for d in data_list]

    ac = np.mean(ac, axis=0)
    tau = fit_exponential(data_ac_ticks, ac)
    return tau