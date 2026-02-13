from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt

def generate_transition_matrix(
    x, y,
    box_length, bins,
    discount_factor, df_damp, threshold,
    sigma, subsample=1, plot=False
):
    # subsample if original FPS is too high
    pos = np.stack([x[::subsample], y[::subsample]], axis=-1)

    # assign bins to the position (split the env in bins)
    pos_digitized = np.digitize(
        pos, np.linspace(0, box_length, bins+1), right=True
    ) - 1
    pos_digitized = pos_digitized[...,0]*bins + pos_digitized[...,1]

    tm = np.zeros((bins*bins,bins*bins))
    # keep a counter of the number of times a bin is visited
    counter_visited = np.zeros((bins*bins,), dtype=np.int32)

    for i in range(1, len(pos_digitized)):
        if i<threshold: # the number of previous positions is less than the threshold
            window = pos_digitized[:i][::-1]
            prob = np.power(discount_factor, df_damp*np.arange(0, i))
        else:
            window = pos_digitized[i-threshold:i][::-1]
            prob = np.power(discount_factor, df_damp*np.arange(0, threshold))
        
        # normalize the probability so that it sums to 1
        prob /= np.sum(prob)

        current_pos = pos_digitized[i]

        # assign the probability to the transition matrix
        tm[current_pos, :np.max(window)+1] += np.bincount(window, prob)
        # update the counter of the number of times a bin was visited
        counter_visited[current_pos] += 1

    # normalize the transition matrix by the number of times a bin was visited
    counter_visited = np.repeat(counter_visited[None, :], bins**2, axis=0).T
    tm_norm = np.divide(tm, counter_visited, out=np.zeros_like(tm), where=counter_visited!=0)

    # apply 2d gaussian filter on the environment
    tm_norm = tm_norm.reshape(-1, bins, bins)
    tm_norm = gaussian_filter(tm_norm, axes=(-2, -1), sigma=[sigma, sigma])
    tm_norm = tm_norm.reshape(bins, bins, bins, bins)

    if plot:
        plot_transition_matrix(
            tm, tm_norm, counter_visited,
            pos, box_length, bins
        )

    return tm, tm_norm

def generate_tm_loop(
    x_list, y_list,
    box_length, bins,
    discount_factor, df_damp, threshold, atol,
    sigma, subsample=1, plot=False
):
    tm_final = np.zeros((bins, bins, bins, bins))

    tm_occ = np.zeros((bins, bins))

    for x, y in zip(x_list, y_list):
        tm, tm_norm = generate_transition_matrix(
            x, y,
            box_length, bins,
            discount_factor, df_damp, threshold,
            sigma=sigma, subsample=subsample, plot=plot
        )
        # keep a counter of the number of times a cell has valid transition probabilities
        tm_norm_occ = np.sum(tm_norm, axis=(-2,-1))
        assert np.allclose(tm_norm_occ[tm_norm_occ>0], 1, atol=atol)
        
        tm_final += tm_norm
        tm_occ += tm_norm_occ

    # normalize the transition matrix by the number of times a cell has valid transition probabilities
    tm_final = np.divide(
        tm_final.T, tm_occ.T,
        out=np.zeros_like(tm_final.T),
        where=tm_occ.T != 0
    ).T

    # we are working with probability distribution, the sum should always be 1
    tmp = np.sum(tm_final, axis=(-2,-1))
    assert np.allclose(tmp[tmp>0], 1, atol=atol)

    return tm_final, tm_occ

def reorder_transition_matrix(tm_3d, bins):
    tm_reord = np.zeros((bins*bins, bins*bins))
    for x in range(bins):
        for y in range(bins):
            tm_reord[x*bins:x*bins+bins, y*bins:y*bins+bins] = tm_3d[x, y]
    return tm_reord

def fold_transition_matrix(tm_3d, tm_reord, bins, half_idx):
    
    tm_occ = np.sum(tm_3d, axis=(-2,-1)).copy()
    tm_occ = np.repeat(np.repeat(tm_occ, bins, axis=0), bins, axis=1)

    tm_occ_half = (np.flip(tm_occ, axis=1)+tm_occ)[:,:half_idx]
    tm_reord_half = (np.flip(tm_reord, axis=1)+tm_reord)[:,:half_idx].copy()
        
    tm_occ_quarter = (np.flip(tm_occ_half, axis=0)+tm_occ_half)[:half_idx,:]
    tm_reord_quarter = (np.flip(tm_reord_half, axis=0)+tm_reord_half)[:half_idx,:]
    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'invalid value encountered in divide')
        tm_reord_quarter = np.divide(tm_reord_quarter, tm_occ_quarter)
    tm_reord_quarter[np.isnan(tm_reord_quarter)] = 0.0
    
    return tm_reord_quarter, tm_occ_quarter, tm_reord_half, tm_occ_half

def reord_folded_transition_matrix(tm_reord_quarter, bins):
    tm_fold = np.zeros((int(bins/2), int(bins/2), bins, bins))
    for x in range(int(bins/2)):
        for y in range(int(bins/2)):
            tm_fold[x, y] = tm_reord_quarter[x*bins:x*bins+bins, y*bins:y*bins+bins]
    return tm_fold

def plot_transition_matrix(tm, tm_norm, counter_visited, pos, box_length, bins):
    fig, ax = plt.subplots(2, 3, figsize=(15,12))

    xy_to_idx = lambda x, y: x*bins + y

    ax[0,0].axis('off')
    ax[0,2].axis('off')

    ax[0,1].scatter(pos[:,0], pos[:,1], s=1)
    ticks = np.linspace(0, box_length, bins+1)
    ax[0,1].set_xticks(ticks, labels=[np.round(t, 2) for t in ticks], rotation=90)
    ax[0,1].set_yticks(ticks, labels=[np.round(t, 2) for t in ticks])
    for t in ticks:
        ax[0,1].hlines(t, 0, box_length, color='gray', alpha=0.5)
        ax[0,1].vlines(t, 0, box_length, color='gray', alpha=0.5)

    counter_plot = counter_visited[np.arange(bins**2), 0].reshape(bins, bins).T
    ax[1,0].imshow(counter_plot, origin='lower', cmap='coolwarm')
    # Loop over data dimensions and create text annotations.
    for i in range(len(counter_plot)):
        for j in range(len(counter_plot)):
            ax[1,0].text(j, i, f"{counter_plot[i, j]}", ha="center", va="center", color="w")
    ax[1,0].set_title(f"{np.min(counter_visited)} -> {np.max(counter_visited)}")
    
    tm_plot = tm[xy_to_idx(2, 2)].reshape(bins,bins).T
    ax[1,1].imshow(tm_plot, origin='lower', cmap='coolwarm')
    ax[1,1].set_title(f"{np.min(tm_plot):.2f} -> {np.max(tm_plot):.2f}")
    for i in range(len(tm_plot)):
        for j in range(len(tm_plot)):
            ax[1,1].text(j, i, f"{tm_plot[i, j]:.0f}", ha="center", va="center", color="w")

    tm_plot = tm_norm[2, 2, ...].T
    ax[1,2].imshow(tm_plot, origin='lower', cmap='coolwarm')
    ax[1,2].set_title(f"{np.min(tm_plot):.2f} -> {np.max(tm_plot):.2f}")
    for i in range(len(tm_plot)):
        for j in range(len(tm_plot)):
            ax[1,2].text(j, i, f"{tm_plot[i, j]:.2f}", ha="center", va="center", color="w")
    plt.show()

def plot_all_transition_matrices(
    tm_reord, tm_reord_half, tm_reord_quarter,
    tm_occ, tm_occ_half, tm_occ_quarter, 
    bins, half_idx
):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(tm_reord, cmap='jet')
    if len(tm_occ[tm_occ<1e-5])>0 : ax[0].imshow(tm_occ, cmap='gray', alpha=0.1)
    ax[0].set_xticks(np.linspace(0, bins**2, bins+1))
    ax[0].set_yticks(np.linspace(0, bins**2, bins+1))
    ax[0].vlines(half_idx, 0, bins**2-1, color='red', linewidth=0.5)
    ax[0].arrow(half_idx+5, half_idx, -10, 0, color='red', head_width=2, length_includes_head=True)
    ax[0].set_title("Original TM")

    ax[1].imshow(tm_reord_half, cmap='jet')
    if len(tm_occ[tm_occ<1e-5])>0 : ax[1].imshow(tm_occ_half, cmap='gray', alpha=0.2)
    ax[1].set_xticks(np.linspace(0, half_idx, bins//2+1))
    ax[1].set_yticks(np.linspace(0, bins**2, bins+1))
    ax[1].hlines(half_idx, 0, half_idx-1, color='red', linewidth=0.5)
    ax[1].arrow(half_idx//2, half_idx+5, 0, -10, color='red', head_width=2, length_includes_head=True)
    ax[1].set_title("Folded TM")

    ax[2].imshow(tm_reord_quarter, cmap='jet')
    if len(tm_occ[tm_occ<1e-5])>0 : ax[2].imshow(tm_occ_quarter, cmap='gray', alpha=0.3)
    ax[2].set_xticks(np.linspace(0, half_idx, bins//2+1))
    ax[2].set_yticks(np.linspace(0, half_idx, bins//2+1))
    ax[2].set_title("Fold-folded TM")

    return fig