import numpy as np
from scipy.ndimage import gaussian_filter
from pycircstat import rayleigh

from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon

def spatial_info(rate: np.ndarray, occ: np.ndarray, method="bit_sec_hertz") -> float:
    # ref Skaggs, Markus 1996
    # Climer paper
    # Sousa paper
    # repo
    # https://github.com/DombeckLab/infoTheory/blob/master/smgmMI.m
    # https://github.com/kevin-allen/spatialInfoScore/blob/main/SpatialInfoScore.ipynb
    # https://github.com/tortlab/spatial-information-metrics/blob/master/info_metrics.m
    # I do not like it it needs to be re-written
    if method not in ["bit_sec", "bit_sec_hertz"]:
        raise ValueError(f"method {method} not recognized")
    no_occ = np.logical_or(occ == 0, np.isnan(occ))
    _rate = rate[~no_occ]
    _occ = occ[~no_occ]
    # turn occ in a probability
    duration = np.sum(_occ)
    _occ_prob = _occ / duration

    if np.isnan(_rate).all() : return 0

    _rate_mean = np.sum(_rate*_occ) / duration
    mask = _rate > 0
    SI = np.sum(
        _occ_prob[mask] * _rate[mask] * np.log2(_rate[mask] / _rate_mean)
    )
    if method == "bit_sec_hertz":
        SI /= _rate_mean
    return SI

def rayleigh_test(bins: np.array, data: np.array, n_samples: int):
    """
    Performs a Rayleigh test on the given data.

    Args:
    bins: (n_bins) An array representing the bins for the data. For example from -pi to pi.
    data: (n_bins) An array containing activity for each directional bin.
    n_samples: The number of samples. Equals n_bins.

    Returns:
    The result of the Rayleigh test.
    """
    return rayleigh(
        bins,
        # rayleigh needs integer number of occurences, so we
        # multiply probabilities by 10*n_samples and round to int
        w=np.rint(n_samples*10*data/np.sum(data)),
        d=2*np.pi/n_samples
    )[0]

def resultant_vector(alpha_bins, nanrobust, axis=None, w=None, d=0):
    """
    Calculate the resultant vector of a given set of angles.

    Args:
    alpha_bins (ndarray): Array of angles.
    nanrobust (bool): Flag indicating whether to handle NaN values robustly.
    axis (int or tuple of ints, optional): Axis or axes along which the sum is computed. Default is None.
    w (ndarray, optional): Array of weights. Default is None.
    d (float, optional): Known spacing between angles. Default is 0.
    """

    if w is None:
        w = np.ones(alpha_bins.shape)

    # weight sum of cos & sin angles
    t = w * np.exp(1j * alpha_bins)
    if nanrobust:
        r = np.nansum(t, axis=axis)
        w[np.isnan(t)] = 0
    else:
        r = np.sum(t, axis=axis)

    r_angle = np.angle(r)

    if np.sum(w, axis=axis) == 0 : return 0, 0

    # obtain length
    r = np.abs(r) / np.sum(w, axis=axis)

    # for data with known spacing, apply correction factor to correct
    # for bias in the estimation of r
    # (see Biostatistical Analysis, J. H. Zar, p. 601, equ. 26.16)
    if d != 0:
        r *= d / 2 / np.sin(d / 2)

    return r, r_angle

def kl_divergence(p, q, eps=1e-10):
    # clip values to avoid log(0)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return np.nansum(p * np.log(p / q))

def get_jensen_shannon(
    rm1:np.array, rm2:np.array, axis:tuple=0,
):
    # first mask elements close to zero
    mask_include = np.logical_and(
        rm1.sum(axis=axis) > 0,
        rm2.sum(axis=axis) > 0
    )
    js = jensenshannon(rm1, rm2, axis=axis)
    if not np.any(mask_include):
        return np.nan
    
    return js[mask_include] if len(js.shape) > 0 else js

def get_spatial_correlation(
    rm1:np.array, rm2:np.array,
    min_included_value:float=1e-5, min_bins:int=2,
    mask:np.array=None, abs:bool=False,
    return_pvalue:bool=True, normalize:bool=False
):
    """
    Calculate the spatial correlation between two 2D arrays, rate maps.

    Args:
    rm1 (np.array): (n_samples, n_samples) The first rate map.
    rm2 (np.array): (n_samples, n_samples) The second rate map.
    min_included_value (float, optional): Minimum value to include in the correlation calculation. Defaults to 1e-5.
    min_bins (int, optional): Minimum number of bins to include in the correlation calculation. Defaults to 2.
    mask (np.array, optional): (n_samples, n_samples) A mask to apply to the rate maps. Defaults to None.
    abs (bool, optional): Whether to return the absolute value of the correlation. Defaults to False.

    Returns:
    tuple(float, float): The Pearson spatial correlation between rm1 and rm2 and its p-value.
    """

    if normalize:
        rm1 = (rm1 - np.nanmin(rm1)) / (np.nanmax(rm1) - np.nanmin(rm1))
        rm2 = (rm2 - np.nanmin(rm2)) / (np.nanmax(rm2) - np.nanmin(rm2))

    if not (mask is None) and (len(mask.shape) != 2):
        raise ValueError(f'Incorrect mask shape {mask.shape}')
    if not (mask is None) and (mask.shape[0] != rm1.shape[0] or mask.shape[1] != rm1.shape[1]):
        raise ValueError(f'Mask shape {mask.shape} does not match with map shape {rm1.shape}')
    
    if np.allclose(rm1, np.mean(rm1), atol=min_included_value) or\
    np.allclose(rm2, np.mean(rm2), atol=min_included_value):
        # Pearson's R, p-value
        return (np.nan, np.nan) if return_pvalue else np.nan
    
    # first mask elements close to zero
    mask_include = np.logical_not(np.logical_or(
        np.logical_or(rm1 < min_included_value, np.isnan(rm1)),
        np.logical_or(rm2 < min_included_value, np.isnan(rm2))
    ))
    if mask is not None:
        mask_include = np.logical_and(mask_include, mask)

    if np.sum(mask_include) < min_bins:
        # Pearson's R, p-value
        return (np.nan, np.nan) if return_pvalue else np.nan
    
    r = tuple(pearsonr(rm1[mask_include], rm2[mask_include]))
    if abs:
        r = (np.abs(r[0]), r[1])

    if np.isnan(r[0]):
        r = (0, r[1])
    return r if return_pvalue else r[0]


def bin_data(var_to_bin_by, n_bins, limits, var_to_bin=[]):
    """
    Creates an N-dimensional histogram of var_to_bin_by using bins of size bin_size.
    If var_to_bin is provided, it creates a weighted histogram using var_to_bin as the weights.
    For example, if var_to_bin is spikes and var_to_bin_by is x-y position, it creates a
    spatial histogram of spike counts.
    
    Parameters
    ----------
    var_to_bin_by : array-like, shape (n_samples, n_dim)
        The data to be binned. Each row represents a sample, and each column represents a dimension.
    n_bins
    limits : list of tuples, length n_dim
        The lower and upper limits for each dimension. Each tuple should contain the lower and upper limit
        for the corresponding dimension.
    var_to_bin : 1D array-like, optional
        The weights for the counts in the histogram. If provided, the histogram will be weighted.
    
    Returns
    -------
    binned_data : array-like
        The N-dimensional histogram of var_to_bin_by.
    """
    
    n_dim = var_to_bin_by.shape[-1]
    if np.size(n_bins) < n_dim:
        n_bins = np.repeat(n_bins, n_dim)
    
    bins = []
    if n_dim == 1:
        bins.append(np.linspace(limits[0], limits[1], n_bins+1))
    else:
        for i in range(n_dim):
            bins.append(np.linspace(limits[i][0], limits[i][1], n_bins[i]+1))
        
    if len(var_to_bin) == 0:
        hst = np.histogramdd(var_to_bin_by, bins=bins)
    else:
        hst = np.histogramdd(var_to_bin_by, bins=bins, weights=var_to_bin, density=False)

    return hst[0]


def bin_data_size(var_to_bin_by, bin_size, limits, var_to_bin=[]):
    """
    Creates an N-dimensional histogram of var_to_bin_by using bins of size bin_size.
    If var_to_bin is provided, it creates a weighted histogram using var_to_bin as the weights.
    For example, if var_to_bin is spikes and var_to_bin_by is x-y position, it creates a
    spatial histogram of spike counts.
    
    Parameters
    ----------
    var_to_bin_by : array-like, shape (n_samples, n_dim)
        The data to be binned. Each row represents a sample, and each column represents a dimension.
    bin_size : scalar or tuple
        The size of the bins. If a scalar is provided, it is used as the bin size for all dimensions.
        If a tuple is provided, it should have the same length as the number of dimensions in var_to_bin_by,
        and each element of the tuple represents the bin size for the corresponding dimension.
    limits : list of tuples, length n_dim
        The lower and upper limits for each dimension. Each tuple should contain the lower and upper limit
        for the corresponding dimension.
    var_to_bin : 1D array-like, optional
        The weights for the counts in the histogram. If provided, the histogram will be weighted.
    
    Returns
    -------
    binned_data : array-like
        The N-dimensional histogram of var_to_bin_by.
    """
    
    n_dim = var_to_bin_by.shape[-1]
    if np.size(bin_size) < n_dim:
        bin_size = np.repeat(bin_size, n_dim)
    
    bins = []
    for i in range(n_dim):
        bins.append(np.arange(limits[i][0], limits[i][1] + bin_size[i], bin_size[i]))
        
    if len(var_to_bin) == 0:
        hst = np.histogramdd(var_to_bin_by, bins=bins)
    else:
        hst = np.histogramdd(var_to_bin_by, bins=bins, weights=var_to_bin, density=False)
        
    return hst[0].T


def get_smooth_polar_map(
    rates:np.array, thetas:np.array,
    limits=(-np.pi, np.pi), n_bins:int=60, sigma:float=5
):
    """
    Calculates the smooth polar map for each neuron based on the given firing rates and thetas.

    Args:
    rates (np.array): (samples, neurons) Array of firing rates for each neuron.
    thetas (np.array): (samples,) Array of head direction points.
    limits (tuple, optional): Limits of the hd space. Defaults to (-np.pi, np.pi).
    bin_size (float, optional): Size of each bin in the hd space. Defaults to 6 degrees.
    sigma (float, optional): Standard deviation of the circular filter. Defaults to 5.

    Returns:
    np.array: Array of smooth rate maps for each neuron.
    """
    
    occupancy = bin_data(thetas[:, None], n_bins=n_bins, limits=limits)
    occupancy_smoothed = gaussian_filter(occupancy, sigma, mode='wrap')

    n_cells = rates.shape[1]
    polar_maps = np.empty(shape=(n_cells,occupancy.shape[0]))
    for i in range(n_cells):
        activations = bin_data(thetas[:, None], n_bins=n_bins, limits=limits, var_to_bin=rates[:,i])
        polar_maps[i] = np.divide(
            gaussian_filter(activations, sigma, mode='wrap'), occupancy_smoothed,
            where=occupancy!=0, out=np.nan*np.ones_like(activations)
        )

    return polar_maps, occupancy_smoothed


def get_smooth_rate_map(
    rates:np.array, trajectory:np.array,
    limits=[(0, 0.5),(0, 0.5)], n_bins:int=25, sigma:float=2
):
    """
    Calculates the smooth rate map for each neuron based on the given firing rates and trajectory.

    Args:
    rates (np.array): (samples, neurons) Array of firing rates for each neuron.
    trajectory (np.array): (samples, 2) Array of trajectory points.
    limits (list, optional): Limits of the trajectory space. Defaults to [(0, 0.5),(0, 0.5)].
    bin_size (float, optional): Size of each bin in the trajectory space. Defaults to 0.01.
    sigma (float, optional): Standard deviation of the Gaussian filter. Defaults to 2.

    Returns:
    np.array: Array of smooth rate maps for each neuron.
    """
    
    occupancy = bin_data(trajectory, n_bins=n_bins, limits=limits)
    occupancy_smoothed = gaussian_filter(occupancy,sigma)

    n_cells = rates.shape[1]
    rate_maps = np.empty(shape=(n_cells,occupancy.shape[0],occupancy.shape[1]))
    for i in range(n_cells):
        activations = bin_data(trajectory, n_bins=n_bins, limits=limits, var_to_bin=rates[:,i])
        rate_maps[i,:,:] = np.divide(
            gaussian_filter(activations,sigma), occupancy_smoothed,
            where=occupancy!=0, out=np.nan*np.ones_like(activations)
        )
    
    # move neurons index at last axis
    rate_maps = np.moveaxis(rate_maps, 0, -1)
    # swap x and y coordinates and flip y so that they appear correctly in the image
    rate_maps = np.transpose(rate_maps, (1, 0, 2))
    rate_maps = np.flip(rate_maps, axis=0)

    return rate_maps, occupancy_smoothed

