import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def plot_trajectory_heatmap(
    exp_dir:str, positions:np.array, n_bins:int, appendix:str=''
):
    """
    Plots trajectory heatmaps based on the given positions and thetas.

    Args:
        exp_dir (str): Directory to save the heatmap image.
        positions (numpy.ndarray): (samples, 2) Array of positions.
        n_bins (int): Number of positional bins.
        appendix (str, optional): Appendix to add to the image filename. Defaults to ''.
    """
    
    x = positions[:,0]
    y = positions[:,1]

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=n_bins)
    heatmap = gaussian_filter(heatmap, sigma=1.5)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.figure(figsize=(4,4))
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='jet')
    plt.axis('off')
    cbar = plt.colorbar(ticks=np.linspace(heatmap.min(), heatmap.max(), 10, endpoint=True))
    cbar.ax.tick_params(labelsize=10)
    plt.tight_layout()
    
    plt.savefig(os.path.join(exp_dir, f'pos_heatmap{appendix}.png'), bbox_inches='tight')
    plt.close()
