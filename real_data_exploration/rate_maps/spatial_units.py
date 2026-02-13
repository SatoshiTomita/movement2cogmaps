import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from utils.metrics import get_smooth_polar_map, get_smooth_rate_map, get_spatial_correlation
from utils.metrics import spatial_info, resultant_vector, kl_divergence
from utils.spatial_fields import detect_fields

SMOOTH_SIGMA = .75




class RateMaps():
    def __init__(
        self, positions:np.array, n_samples_pos:int,
        env_dim:float, sigma:float=SMOOTH_SIGMA
    ):
        """
        Args:
        positions (np.array): (samples, 2) Array of positions.
        n_samples_pos (int): Number of samples for positions.
        env_dim (float): Dimension of the environment.
        sigma (float, optional): Standard deviation for smoothing. Defaults to SMOOTH_SIGMA.
        """
        
        self.positions = positions
        self.n_samples_pos = n_samples_pos
        self.sigma = sigma
        self.env_dim = env_dim
        self.bin_size = env_dim/n_samples_pos

    def calculate_rate_maps(self, latent_activity:np.array):
        """
        Args:
        latent_activity (np.array): (samples, n_units)
            Activity rate for each unit over samples.

        Returns:
        rate_maps (np.array): (n_units, n_samples_pos, n_samples_pos)
            Array of rate maps, one for each unit.
        """
        rate_maps = get_smooth_rate_map(
            latent_activity,
            self.positions,
            limits = [(0, self.env_dim), (0, self.env_dim)],
            n_bins = self.n_samples_pos,
            sigma = self.sigma
        )
        rate_maps = np.moveaxis(rate_maps, -1, 0)

        return rate_maps

    def calculate_rate_maps_vs_hd(self, latent_activity:np.array, n_rotations:int, indices:np.array):
        """
        Calculate the rate maps for each head direction.

        Args:
        latent_activity (np.array): (samples, n_units)
            Activity rate for each unit over samples.
        n_rotations (int): Number of directional samples.
        indices (np.array): (samples, 1) Directionality index for each sample.
        """
        rate_maps_hd = []

        for idx in range(1, n_rotations+1):
            rate_maps_hd.append(
                get_smooth_rate_map(
                    latent_activity[indices==idx],
                    self.positions[indices==idx],
                    limits = [(0, self.env_dim), (0, self.env_dim)],
                    n_bins = self.n_samples_pos,
                    sigma = self.sigma
                )
            )
            
        rate_maps_hd = np.stack(rate_maps_hd, axis=0)
        rate_maps_hd = np.moveaxis(rate_maps_hd, -1, 0)
        
        return rate_maps_hd

    def rate_maps_stability(self, rate_maps_1:np.array, rate_maps_2:np.array):
        """
        Comparison between two sets of place cells.

        Args:
        rate_maps_1 (np.array): (n_units, n_samples_pos, n_samples_pos) Array of neurons' activities for the first trajectory.
        rate_maps_2 (np.array): (n_units, n_samples_pos, n_samples_pos) Array of neurons' activities for the seconds trajectory.

        Returns:
        (np.array): (n_units, 1) The correlation between the two sets of place cells.
        """
        rate_maps_1 = np.nan_to_num(rate_maps_1, copy=True)
        rate_maps_2 = np.nan_to_num(rate_maps_2, copy=True)

        correlations = []

        for i in range(rate_maps_1.shape[0]):
            corr, _ = get_spatial_correlation(
                rate_maps_1[i], rate_maps_2[i]
            )
            correlations.append(corr)

        return np.array(correlations)

    def rate_maps_hd_stability(self, rate_maps:np.array, rate_maps_hd:np.array):
        """
        Comparison between two sets of place cells.

        Args:
        rate_maps (np.array): (n_units, n_samples_pos, n_samples_pos) Array of neurons' activities.
        rate_maps_hd (np.array): (n_units, n_rotations, n_samples_pos, n_samples_pos) Array of neurons' activities for each head direction.
        
        Returns:
        (np.array): (n_units, 1) The correlation for each rate map.
        """
        correlations = np.zeros(rate_maps.shape[0])
        for h in range(rate_maps_hd.shape[1]):
            rm_hd = rate_maps_hd[:, h, ...]
            
            corr = [
                get_spatial_correlation(rate_maps[i], rm_hd[i])[0]\
                for i in range(rate_maps.shape[0])
            ]
            correlations += corr

        correlations /= rate_maps_hd.shape[1]
        return correlations

    def rate_maps_field_detection(self, rate_maps:np.array, rate_maps_1:np.array, rate_maps_2:np.array):
        """
        Detect the fields for the rate maps.

        Args:
        rate_maps (np.array): (n_units, n_samples_pos, n_samples_pos) Array of rate maps.
        rate_maps_1 (np.array): (n_units, n_samples_pos, n_samples_pos) Array of neurons' activities for the first trajectory.
        rate_maps_2 (np.array): (n_units, n_samples_pos, n_samples_pos) Array of neurons' activities for the seconds trajectory.

        Returns:
        (np.array): (n_units, 1) The number of fields for each rate map.
        (List[List[[np.array]]): The list of fields for each rate map.
        """
        rate_maps = np.nan_to_num(rate_maps, copy=True)
        rate_maps_1 = np.nan_to_num(rate_maps_1, copy=True)
        rate_maps_2 = np.nan_to_num(rate_maps_2, copy=True)

        n = rate_maps.shape[-1]

        rate_maps_min = np.moveaxis(
            np.tile(np.min(rate_maps, axis=(1,2)), (n, n, 1)), -1, 0
        )
        rate_maps_max = np.moveaxis(
            np.tile(np.max(rate_maps, axis=(1,2)), (n, n, 1)), -1, 0
        )
        rate_maps_norm = np.divide(
            (rate_maps - rate_maps_min),
            (rate_maps_max - rate_maps_min),
            where=(rate_maps_max - rate_maps_min)!=0,
            out=rate_maps
        )

        rm_fields = []

        params = {
            'base_threshold': 0.1,
            'threshold_step': 0.05,
            'primary_filter_kwargs': {
                'min_bins': 10, 'min_peak_value': 0.5
            },
            'secondary_filter_kwargs': {
                'min_stability': 0.25, 'max_relative_bins': 0.5,
                'stability_kwargs': {'min_included_value': 0.01, 'min_bins': 5}
            }
        }
        for rm, rm1, rm2 in zip(rate_maps_norm, rate_maps_1, rate_maps_2):
            fields = detect_fields(
                rm, (rm1, rm2), **params
            )
            rm_fields.append(fields)

        n_fields = np.array([len(f) for f in rm_fields])

        return n_fields, rm_fields

    def rm_flipped(self, rate_maps:np.array, filter_indices:np.array=None):
        """
        Calculate the rate maps by flipping it twice along the center.

        Args:
        rate_maps (np.array): (n_units, n_samples_pos, n_samples_pos) Array of rate maps.
        filter_indices (np.array): (n_units) Indices to filter the rate maps. Defaults to None, no filtering.

        Returns:
        (np.array): (n_units, n_samples_pos/2, n_samples_pos/2) Rate maps flipped.
        """
        rate_maps = np.nan_to_num(rate_maps, copy=True)

        half_idx = int(np.ceil(self.n_samples_pos/2))

        rms = rate_maps if filter_indices is None else rate_maps[filter_indices, ...]

        rate_maps_half = (
            np.flip(rms, axis=1)+rms
        )[:, :half_idx, :] / 2

        rate_maps_quarter = (
            np.flip(rate_maps_half, axis=2)+rate_maps_half
        )[:, :, :half_idx] / 2

        return rate_maps_quarter

    def rm_field_flipped(self, rm_fields:list, filter_indices:np.array=None):
        """
        Calculate the rate maps field by flipping it twice along the center.

        Args:
        rm_fields (List): List of rate maps fields.
        filter_indices (np.array): (n_units) Indices to filter the rate maps. Defaults to None, no filtering.

        Returns:
        (np.array): (n_units, n_samples_pos/2, n_samples_pos/2) rate maps field flipped. 
        """
        half_idx = int(np.ceil(self.n_samples_pos/2))

        fields = rm_fields if filter_indices is None else [f for i, f in enumerate(rm_fields) if i in filter_indices]

        rm_fields = np.array([np.nansum(f, axis=0) for f in fields if f])
        rm_fields[rm_fields>0] = 1

        # if there aren't any fields, just return empty array
        if len(rm_fields.shape) < 2 : return np.array([])

        rm_fields_half = (
            np.flip(rm_fields, axis=1)+rm_fields
        )[:, :half_idx, :] / 2

        rm_fields_quarter = (
            np.flip(rm_fields_half, axis=2)+rm_fields_half
        )[:, :, :half_idx] / 2

        return rm_fields_quarter

    def calculate_metrics(self, rate_maps:np.array, occ:np.array=None):
        """
        Calculate the spatial information, and KL divergence for the rate maps.
        
        Args:
        rate_maps (np.array): (n_units, n_samples_pos, n_samples_pos) Array of rate maps.
        occ (np.array): (n_samples_pos, n_samples_pos) Array of occupancy. Defaults to None: uniform.

        Returns:
        si (np.array): (n_units, 1) Array of spatial information.
        kld (np.array): (n_units, 1) Array of KL divergence.
        ss (np.array): (n_units, 1) Array of spatial sparsity.
        """
        rate_maps = np.nan_to_num(rate_maps, copy=True)

        n = self.n_samples_pos
        rate_maps_min = np.moveaxis(
            np.tile(np.min(rate_maps, axis=(1,2)), (n, n, 1)), -1, 0
        )
        rate_maps_max = np.moveaxis(
            np.tile(np.max(rate_maps, axis=(1,2)), (n, n, 1)), -1, 0
        )
        rate_maps_norm = np.divide(
            (rate_maps - rate_maps_min),
            (rate_maps_max - rate_maps_min),
            where=(rate_maps_max - rate_maps_min)!=0,
            out=rate_maps
        )
        uniform_rate_map = np.ones_like(rate_maps_norm[0])/(n**2)
        if occ is None : occ = uniform_rate_map

        si = [spatial_info(rm, occ) for rm in rate_maps_norm]

        return np.array(si)


class RateMapsPlotter():
    def __init__(self, exp_dir:str, save_figures:bool=True):
        """
        Args:
        exp_dir (str): Path to the experiment directory.
        save_figures (bool): Flag to indicate if the figures should be saved. Defaults to True.
        """
        
        self.exp_dir = exp_dir
        self.save_figures = save_figures

    def average_rate_map(self, rate_maps:np.array):
        """
        Plot the average rate map.
        
        Args:
        rate_maps (np.array): (n_units, n_samples_pos, n_samples_pos) Array of rate maps.
        """
        plt.figure()
        plt.imshow(np.mean(rate_maps, axis=0), cmap='jet')
        plt.colorbar()
        plt.axis('off')
        if self.save_figures:
            plt.savefig(os.path.join(self.exp_dir, 'average_rate_map.png'), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def rate_maps(
        self, ncols:int, rate_maps:np.array, si:np.array,
        unit_names:str=None, appendix:str=''
    ):
        """
        Plot the rate maps.
        
        Args:
        ncols (int): Number of columns for the plot.
        rate_maps (np.array): (n_units, n_samples_pos, n_samples_pos) Array of rate maps.
        si (np.array): (n_units, 1) Array of spatial information.
        unit_names (str, optional): Names for the units. Defaults to None.
        appendix (str, optional): Appendix for the file name. Defaults to ''.
        """
        nrows = int(np.ceil(rate_maps.shape[0]/ncols))
        _, axs = plt.subplots(nrows, ncols, figsize=(2.5*ncols,3*nrows))
        if unit_names is None:
            unit_names = list(range(rate_maps.shape[0]))

        for idx in range(nrows*ncols):
            ax = axs.flat[idx]
            ax.set_xticks([])
            ax.set_yticks([])

            if idx >= rate_maps.shape[0] : continue

            rate_map = rate_maps[idx, ...]

            ax.imshow(rate_map, cmap='jet')
            ax.set_title(
                f'Neuron {unit_names[idx]}\nSI={si[idx]:.2f}', fontsize=7
            )

        if self.save_figures:
            plt.savefig(os.path.join(self.exp_dir, f'rate_maps{appendix}.png'), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def rate_maps_comparison(
        self, ncols:int, rms1:np.array, rms2:np.array, rms_stability:np.array, frac:float=.5,
        unit_names:str=None, appendix:str=''
    ):
        """
        Plot the comparison between two sets of rate maps.
        
        Args:
        ncols (int): Number of columns for the plot.
        rms1 (np.array): (n_units, n_samples_pos, n_samples_pos) Array of rate maps for the first trajectory.
        rms2 (np.array): (n_units, n_samples_pos, n_samples_pos) Array of rate maps for the second trajectory.
        rms_stability (np.array): (n_units, 1) Array of correlations between the two sets of rate maps.
        frac (float, optional): Fraction of neurons to plot. Defaults to half.
        """
        random_indices = np.sort(np.random.choice(
            np.arange(rms1.shape[0], dtype=int),
            size=int(np.ceil(rms1.shape[0]*frac)), replace=False
        ))
        rms1 = rms1[random_indices, ...].copy()
        rms2 = rms2[random_indices, ...].copy()
        corr = rms_stability[random_indices].copy()

        if unit_names is None:
            unit_names = list(random_indices)

        nrows = int(np.ceil(rms1.shape[0]/ncols))
        fig = plt.figure(figsize=(2*ncols,1.5*nrows))
        outer = gridspec.GridSpec(nrows, ncols)

        for i in range(nrows*ncols):
            inner = gridspec.GridSpecFromSubplotSpec(
                1, 2, subplot_spec=outer[i], wspace=0.05, hspace=0.05
            )
            ax = plt.Subplot(fig, outer[i])
            ax.axis('off')

            if i >= rms1.shape[0]:
                continue

            rm1 = rms1[i, ...]
            rm2 = rms2[i, ...]
            ax.set_title(f"Neuron {unit_names[i]}\nCorr {corr[i]:.2f}", fontsize=7)

            fig.add_subplot(ax)

            ax = fig.add_subplot(inner[0])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(rm1, cmap='jet')

            ax = fig.add_subplot(inner[1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(rm2, cmap='jet')

        if self.save_figures:
            plt.savefig(os.path.join(self.exp_dir, f'rate_maps_stability{appendix}.png'), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def rate_maps_vs_hd8(
        self, ncols:int, rate_maps:np.array, rate_maps_hd:np.array,
        rm_hd_stability:np.array, polar_maps:np.array, thetas_ticks,
        frac:float=.5, unit_names:str=None, appendix:str=''
    ):
        """
        Plot the rate maps for each head direction.
        
        Args:
        ncols (int): Number of columns for the plot.
        rate_maps (np.array): (n_units, n_samples_pos, n_samples_pos) Array of rate maps.
        rate_maps_hd (np.array): (n_units, n_rotations, n_samples_pos, n_samples_pos) Array of rate maps for each head direction.
        rm_hd_stability (np.array): (n_units, 1) Array of correlations for each rate map.
        frac (float, optional): Fraction of neurons to plot. Defaults to half.
        """
        random_indices = np.sort(np.random.choice(
            np.arange(rate_maps.shape[0], dtype=int),
            size=int(rate_maps.shape[0]*frac), replace=False
        ))
        rms = rate_maps[random_indices, ...].copy()
        rms_hd = rate_maps_hd[random_indices, ...].copy()
        rm_hd_s = rm_hd_stability[random_indices].copy()

        if unit_names is None:
            unit_names = list(random_indices)

        nrows = int(np.ceil(rms.shape[0]/ncols))

        fig = plt.figure(figsize=(5*ncols,3*nrows))
        outer = gridspec.GridSpec(nrows, ncols)

        for i in range(nrows*ncols):
            inner_place_hd = gridspec.GridSpecFromSubplotSpec(
                1, 2, subplot_spec=outer[i], wspace=0.05, hspace=0.05, width_ratios=[2.5, 1]
            )
            ax = fig.add_subplot(outer[i])
            ax.axis('off')
            if i >= rms.shape[0]:
                continue
            ax.set_title(f"Neuron {unit_names[i]}, corr {rm_hd_s[i]:.2f}", fontsize=7)

            inner_3by3 = gridspec.GridSpecFromSubplotSpec(
                3, 3, subplot_spec=inner_place_hd[0], wspace=0.05, hspace=0.05
            )
            ax = fig.add_subplot(inner_place_hd[0])
            ax.set_axis_off()

            rm_hd = rms_hd[i, ...]
            # ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
            rot_idx_order = [3, 2, 1, 4, 0, 5, 6, 7]
            delta = 0
            for r in range(3):
                for c in range(3):
                    ax = fig.add_subplot(inner_3by3[r, c])
                    ax.set_axis_off()
                    j = r*3 + c
                    if j==4:
                        ax.imshow(rms[i, ...], cmap='jet')
                        delta = -1
                    else:
                        ax.imshow(rm_hd[rot_idx_order[j+delta]], cmap='jet')

            pm = polar_maps[i]
            ax = fig.add_subplot(inner_place_hd[1], projection='polar')
            ax.plot(
                np.append(thetas_ticks, thetas_ticks[0]),
                np.append(pm, pm[0]),
                lw=2, c='blue',
                # marker='o', ms=5, mfc='red'
            )
            ax.set_xticklabels([]) # remove degrees indication
            ax.set_rticks([max(pm)], labels=[f"{max(pm):.2f}"]) # add intensity indication
            ax.tick_params(axis='y', labelsize=5)
            ax.grid(True)

        if self.save_figures:
            fig.savefig(os.path.join(self.exp_dir, f'rate_maps_hd{appendix}.png'), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def rate_maps_vs_hd4(
        self, ncols:int, rate_maps:np.array, rate_maps_hd:np.array,
        rm_hd_stability:np.array, polar_maps:np.array, thetas_ticks,
        frac:float=.5, unit_names:str=None, appendix:str=''
    ):
        """
        Plot the rate maps for each head direction.
        
        Args:
        ncols (int): Number of columns for the plot.
        rate_maps (np.array): (n_units, n_samples_pos, n_samples_pos) Array of rate maps.
        rate_maps_hd (np.array): (n_units, n_rotations, n_samples_pos, n_samples_pos) Array of rate maps for each head direction.
        rm_hd_stability (np.array): (n_units, 1) Array of correlations for each rate map.
        frac (float, optional): Fraction of neurons to plot. Defaults to half.
        """
        random_indices = np.sort(np.random.choice(
            np.arange(rate_maps.shape[0], dtype=int),
            size=int(rate_maps.shape[0]*frac), replace=False
        ))
        rms = rate_maps[random_indices, ...].copy()
        rms_hd = rate_maps_hd[random_indices, ...].copy()
        rm_hd_s = rm_hd_stability[random_indices].copy()

        if unit_names is None:
            unit_names = list(random_indices)

        nrows = int(np.ceil(rms.shape[0]/ncols))

        fig = plt.figure(figsize=(5*ncols,3*nrows))
        outer = gridspec.GridSpec(nrows, ncols)

        for i in range(nrows*ncols):
            inner_place_hd = gridspec.GridSpecFromSubplotSpec(
                1, 2, subplot_spec=outer[i], wspace=0.05, hspace=0.05, width_ratios=[2.5, 1]
            )
            ax = fig.add_subplot(outer[i])
            ax.axis('off')
            if i >= rms.shape[0]:
                continue
            ax.set_title(f"Neuron {unit_names[i]}, corr {rm_hd_s[i]:.2f}", fontsize=7)

            inner_3by3 = gridspec.GridSpecFromSubplotSpec(
                3, 3, subplot_spec=inner_place_hd[0], wspace=0.05, hspace=0.05
            )
            ax = fig.add_subplot(inner_place_hd[0])
            ax.set_axis_off()

            rm_hd = rms_hd[i, ...]
            # ['E', 'N', 'W', 'S']
            rot_idx_order = [1, 1, 2, 2, 0, 0, 3, 3]
            delta = 0
            for r in range(3):
                for c in range(3):
                    ax = fig.add_subplot(inner_3by3[r, c])
                    ax.set_axis_off()
                    j = r*3 + c
                    if j==4:
                        ax.imshow(rms[i, ...], cmap='jet')
                        delta = -1
                    elif j in [1, 3, 5, 7]:
                        ax.imshow(rm_hd[rot_idx_order[j+delta]], cmap='jet')

            pm = polar_maps[i]
            ax = fig.add_subplot(inner_place_hd[1], projection='polar')
            ax.plot(
                np.append(thetas_ticks, thetas_ticks[0]),
                np.append(pm, pm[0]),
                lw=2, c='blue',
                # marker='o', ms=5, mfc='red'
            )
            ax.set_xticklabels([]) # remove degrees indication
            ax.set_rticks([max(pm)], labels=[f"{max(pm):.2f}"]) # add intensity indication
            ax.tick_params(axis='y', labelsize=5)
            ax.grid(True)

        if self.save_figures:
            fig.savefig(os.path.join(self.exp_dir, f'rate_maps_hd{appendix}.png'), bbox_inches='tight')
            plt.close()
        else:
            plt.show()




class PolarMaps():

    def __init__(
        self, thetas:np.array, n_samples_thet:int, minmax_scaling:bool
    ):
        """
        Args:
        thetas (np.array): (samples) Array of head directions in rads.
        n_samples_thet (int): Number of samples for directions.
        minmax_scaling (bool): Flag to indicate if minmax scaling should be used.
        """
        self.thetas = thetas
        self.n_samples_thet = n_samples_thet

        # create directionality bins and indices from 0 to 2*pi
        bins_thet = np.linspace(0, 2*np.pi, self.n_samples_thet+1)
        self.indices = np.digitize(self.thetas, bins_thet) if self.thetas is not None else None
        # the tick sits in the middle of the bins
        self.thetas_ticks = np.array(
            [np.mean([a, b]) for a, b in zip(bins_thet, bins_thet[1:])]
        )
        self.bins_thet = bins_thet[:-1]

        self.minmax_scaling = minmax_scaling

        self.selected_hd_cells = None

    def get_bins(self):
        return self.bins_thet
    
    def get_thetas_ticks(self):
        return self.thetas_ticks

    def get_indices(self):
        return self.indices

    def calculate_polar_maps(self, latent_activity: np.array):
        """
        Calculate the polar data for the given latent data.

        Args:
        latent_activity (np.array): (n_samples, n_units) The latent data, neurons' activities.

        Returns:
        polar_maps (np.array): (n_units, n_samples_thet) Array of polar maps, one for each unit.
        pm_std (np.array): (n_units, n_samples_thet) Array of standard deviations for polar maps.
        """
        # check if there is at least one data point for each directional bin
        # is_data_available = [np.any(self.indices==idx) for idx in range(1, self.n_samples_thet+1)]
        # if not np.all(is_data_available):
        #     print("There is at least one directional bin without data. Check:", is_data_available)

        # # the latent activity is projected in polar space by taking the
        # # average activity for each directional bin
        polar_maps = get_smooth_polar_map(
            latent_activity,
            self.thetas,
            n_bins=self.n_samples_thet,
            sigma=5
        )
        pm_std = [
            np.std(latent_activity[self.indices==idx], axis=0)\
            for idx in range(1, self.n_samples_thet+1)
        ]
        return polar_maps, np.stack(pm_std, axis=1)
    
    def calculate_polar_maps_vs_place(
        self, latent_activity: np.array, positions:np.array, env_dim: float, n_samples_pos: int
    ):
        """
        Calculate the polar data for the given latent data subdivided by positional bins.

        Args:
        latent_activity (np.array): (n_samples, n_units) The latent data, hidden units' activity.
        positions (np.array): (n_samples, 2) Array of positions.
        env_dim (float): Dimension of the environment.
        n_samples_pos (int): Number of bins for positions.

        Returns:
        np.array: (n_neurons, n_samples_pos, n_samples_pos, n_samples_thet) The latent activity in a polar space and in positional bins.
        """
        indices_pos = np.digitize(positions, bins=np.linspace(0, env_dim, n_samples_pos+1))

        polar_maps_place = np.zeros(
            (n_samples_pos, n_samples_pos, latent_activity.shape[-1], self.n_samples_thet)
        )

        for x_bin, y_bin in product(range(1, n_samples_pos+1), range(1, n_samples_pos+1)):
            idx_pos = np.logical_and(indices_pos[:,0]==x_bin, indices_pos[:,1]==y_bin)
            polar_maps_place[x_bin-1, y_bin-1, ...] =\
                get_smooth_polar_map(
                    latent_activity[idx_pos],
                    self.thetas[idx_pos],
                    n_bins=self.n_samples_thet,
                    sigma=5
                )
        polar_maps_place = np.moveaxis(polar_maps_place, -2, 0)

        return polar_maps_place

    def polar_maps_stability(self, polar_maps_1:np.array, polar_maps_2:np.array):
        """
        Comparison between two sets of head direction cells.

        Args:
        polar_maps_1 (np.array): (n_units, n_samples_thet) Array of neurons' activities for the first trajectory.
        polar_maps_2 (np.array): (n_units, n_samples_thet) Array of neurons' activities for the seconds trajectory.

        Returns:
        (np.array): (n_units, 1) The correlation between the two sets of HD cells.
        """
        correlations = []

        for i in range(polar_maps_1.shape[0]):
            corr, _ = get_spatial_correlation(
                polar_maps_1[i], polar_maps_2[i]
            )
            correlations.append(corr)

        return np.array(correlations)
    
    def polar_maps_place_stability(self, polar_maps:np.array, polar_maps_place:np.array):
        """
        Comparison between two sets of place cells.

        Args:
        polar_maps (np.array): (n_units, n_samples_thet) Array of neurons' activities.
        polar_maps_place (np.array): (n_units, n_samples_pos, n_samples_pos, n_samples_thet) Array of neurons' activities in positional bins.

        Returns:
        (np.array): (n_units, 1) The correlation for each polar map.
        """
        correlations = np.zeros(polar_maps.shape[0])

        for p_row in range(polar_maps_place.shape[1]):
            for p_col in range(polar_maps_place.shape[1]):
                pm_place = polar_maps_place[:, p_row, p_col]
                
                corr = [
                    get_spatial_correlation(polar_maps[i], pm_place[i])[0]\
                    for i in range(polar_maps.shape[0])
                ]
                correlations += corr

        correlations /= (polar_maps_place.shape[1]**2)
        return correlations

    def calculate_metrics(self, polar_maps:np.array):
        """
        Calculate the spatial information, Rayleigh Vector length and KL divergence for the polar maps.
        
        Args:
        polar_maps (np.array): (n_units, n_samples_thet) Array of polar maps.
        
        Returns:
        si (np.array): (n_units, 1) Array of spatial information.
        rvl (np.array): (n_units, 1) Array of RV lengths.
        rvangle (np.array): (n_units, 1) Array of RV angles.
        kld (np.array): (n_units, 1) Array of KL divergence.
        """
        n = polar_maps.shape[-1]

        divisor = np.moveaxis(
            np.tile(polar_maps.sum(axis=1), (n, 1)), -1, 0
        )
        polar_maps_norm = np.divide(
            polar_maps, divisor,
            where=divisor != 0, out=polar_maps
        )
        uniform_polar_map = np.ones_like(polar_maps[0])/self.n_samples_thet

        si = [spatial_info(pm, uniform_polar_map) for pm in polar_maps_norm]
        rv = [
            resultant_vector(
                self.bins_thet,
                nanrobust=False,
                w=pm,
                d=2*np.pi/self.n_samples_thet
            ) for pm in polar_maps_norm
        ]
        kld = [kl_divergence(pm, uniform_polar_map) for pm in polar_maps_norm]

        rvl = [x[0] for x in rv]
        rvangle = [x[1] for x in rv]

        return np.array(si), np.array(rvl), np.array(rvangle), np.array(kld)


class PolarMapsPlotter():
    def __init__(self, exp_dir:str, save_figures:bool=True):
        """
        Args:
        exp_dir (str): Path to the experiment directory.
        """

        self.exp_dir = exp_dir
        self.save_figures = save_figures

    def average_polar_map(self, polar_maps:np.array, thetas_ticks:np.array):
        """
        Plot the average polar map.
        
        Args:
        polar_maps (np.array): (n_units, n_samples_thet) Array of polar maps.
        thetas_ticks (np.array): (n_samples_thet) Array of ticks for the polar maps.
        """
        _, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        polar_map = np.mean(polar_maps, axis=0)
        ax.plot(
            np.append(thetas_ticks, thetas_ticks[0]), # we want to close the circle
            np.append(polar_map, polar_map[0]), # we want to close the circle
            lw=3, c='blue',
            # marker='o', ms=5, mfc='red'
        )
        ax.set_xticklabels([]) # remove degrees indication
        ax.set_rticks([]) # remove intensity indication
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N') # move 0 to the north
        ax.grid(True)
        ax.set_title('Average polar map')
        if self.save_figures:
            plt.savefig(os.path.join(self.exp_dir, 'average_polar_map.png'), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def polar_maps(
        self, ncols:int, polar_maps:np.array, thetas_ticks:np.array,
        kld:np.array, rvangle:np.array, rvl:np.array=None,
        unit_names:str=None, appendix:str=''
    ):
        """
        Plot the polar maps.
        
        Args:
        ncols (int): Number of columns for the plot.
        polar_maps (np.array): (n_units, n_samples_thet) Array of polar maps.
        thetas_ticks (np.array): (n_samples_thet) Array of ticks for the polar maps.
        kld (np.array): (n_units, 1) Array of KL divergence.
        rvl (np.array): (n_units, 1) Array of RV lengths.
        rvangle (np.array): (n_units, 1) Array of RV angles.
        unit_names (str, optional): Names for the units. Defaults to None.
        appendix (str, optional): Appendix for the file name. Defaults to ''.
        """
        nrows = int(np.ceil(polar_maps.shape[0]/ncols))
        _, axs = plt.subplots(
            nrows, ncols, subplot_kw={'projection': 'polar'},
            figsize=(2.5*ncols,3*nrows)
        )
        if unit_names is None:
            unit_names = list(range(polar_maps.shape[0]))

        for idx in range(nrows*ncols):
            ax = axs.flat[idx]

            if idx >= polar_maps.shape[0]:
                ax.set_axis_off()
                continue

            polar_map = polar_maps[idx, ...]
            # dlp_std_n = dlp_std[:, r*ncols+c]
            if np.isnan(polar_map).all(): print("polar map is completely nan")
            ax.plot(
                np.append(thetas_ticks, thetas_ticks[0]), # we want to close the circle
                np.append(polar_map, polar_map[0]), # we want to close the circle
                lw=3, c='blue',
                # marker='o', ms=5, mfc='red'
            )
            _len = rvl[idx] if rvl is not None else np.max(polar_map)
            ax.vlines(
                rvangle[idx], 0, _len, #np.max(polar_map),
                colors='red', lw=2
            )
            # ax.set_rmax(1)
            ax.set_xticklabels([]) # remove degrees indication
            ax.set_rticks([max(polar_map)], labels=[f"{max(polar_map):.2f}"]) # add intensity indication
            ax.tick_params(axis='y', labelsize=9)
            ax.grid(True)
            ax.set_title(
                f'Neuron {unit_names[idx]}'+\
                f'\nKL:{kld[idx]:.1f},R:{rvl[idx]:.1f}', fontsize='xx-small'
            )

        if self.save_figures:
            plt.savefig(os.path.join(self.exp_dir, f'polar_maps{appendix}.png'), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def polar_maps_comparison(
        self, ncols:int, pms1:np.array, pms2:np.array,
        thetas_ticks:np.array, pms_stability:np.array, frac:float=.5,
        unit_names:str=None, appendix:str=''
    ):
        """
        Plot the comparison between two sets of polar maps.
        
        Args:
        ncols (int): Number of columns for the plot.
        pms1 (np.array): (n_units, n_samples_thet) Array of polar maps for the first trajectory.
        pms2 (np.array): (n_units, n_samples_thet) Array of polar maps for the second trajectory.
        thetas_ticks (np.array): (n_samples_thet) Array of ticks for the polar maps.
        pms_stability (np.array): (n_units, 1) Array of correlations between the two sets of polar maps.
        frac (float, optional): Fraction of neurons to plot. Defaults to half.
        """
        random_indices = np.sort(np.random.choice(
            np.arange(pms1.shape[0], dtype=int),
            size=int(pms1.shape[0]*frac), replace=False
        ))
        pms1 = pms1[random_indices, ...].copy()
        pms2 = pms2[random_indices, ...].copy()
        corr = pms_stability[random_indices].copy()

        if unit_names is None:
            unit_names = list(random_indices)

        nrows = int(np.ceil(pms1.shape[0]/ncols))
        fig = plt.figure(figsize=(2.5*ncols,1.5*nrows))
        outer = gridspec.GridSpec(nrows, ncols)

        for i in range(nrows*ncols):
            inner = gridspec.GridSpecFromSubplotSpec(
                1, 2, subplot_spec=outer[i], wspace=0.05, hspace=0.05
            )
            ax = plt.Subplot(fig, outer[i])
            ax.axis('off')

            if i >= pms1.shape[0]:
                continue

            pm1 = pms1[i, ...]
            pm2 = pms2[i, ...]
            ax.set_title(f"Neuron {unit_names[i]}\nCorr {corr[i]:.2f}", fontsize=7)

            ax = fig.add_subplot(inner[0], polar=True)
            ax.plot(
                np.append(thetas_ticks, thetas_ticks[0]), # we want to close the circle
                np.append(pm1, pm1[0]), # we want to close the circle
                lw=3, c='blue',
                # marker='o', ms=5, mfc='red'
            )
            # ax.set_rmax(1)
            ax.set_xticklabels([]) # remove degrees indication
            ax.set_rticks([max(pm1)], labels=[f"{max(pm1):.2f}"]) # add intensity indication
            ax.tick_params(axis='y', labelsize=5)
            ax.set_theta_direction(-1)
            ax.set_theta_zero_location('N') # move 0 to the north
            ax.grid(True)

            ax = fig.add_subplot(inner[1], polar=True)
            ax.plot(
                np.append(thetas_ticks, thetas_ticks[0]), # we want to close the circle
                np.append(pm2, pm2[0]), # we want to close the circle
                lw=3, c='blue',
                # marker='o', ms=5, mfc='red'
            )
            # ax.set_rmax(1)
            ax.set_xticklabels([]) # remove degrees indication
            ax.set_rticks([max(pm2)], labels=[f"{max(pm2):.2f}"]) # add intensity indication
            ax.tick_params(axis='y', labelsize=5)
            ax.set_theta_direction(-1)
            ax.set_theta_zero_location('N') # move 0 to the north
            ax.grid(True)

        if self.save_figures:
            plt.savefig(os.path.join(self.exp_dir, f'polar_maps_stability{appendix}.png'), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def polar_maps_vs_place(
        self, ncols:int, polar_maps_place:np.array,
        n_samples_pos:int, bins_thet:np.array, n_samples_thet:int, thetas_ticks:np.array,
        pm_place_stability:np.array, frac:float=.5,
        unit_names:str=None, appendix:str=''
    ):
        """
        Plot the polar maps subdivided by positional bins.
        
        Args:
        ncols (int): Number of columns for the plot.
        polar_maps_place (np.array): (n_units, n_samples_pos, n_samples_pos, n_samples_thet) Array of polar maps in positional bins.
        n_samples_pos (int): Number of bins for positions.
        bins_thet (np.array): (n_samples_thet) Array of bins for head directions.
        n_samples_thet (int): Number of bins for head directions.
        thetas_ticks (np.array): (n_samples_thet) Array of ticks for the polar maps.
        pm_place_stability (np.array): (n_units, 1) Array of correlations for each polar map.
        frac (float, optional): Fraction of neurons to plot. Defaults to half.
        """
        random_indices = np.sort(np.random.choice(
            np.arange(polar_maps_place.shape[0], dtype=int),
            size=int(polar_maps_place.shape[0]*frac), replace=False
        ))
        pms_place = polar_maps_place[random_indices, ...].copy()
        pm_place_s = pm_place_stability[random_indices].copy()

        if unit_names is None:
            unit_names = list(random_indices)

        nrows = int(np.ceil(pms_place.shape[0]/ncols))

        fig = plt.figure(figsize=(2*ncols,2*nrows))
        outer = gridspec.GridSpec(nrows, ncols)

        for i in range(nrows*ncols):
            inner = gridspec.GridSpecFromSubplotSpec(
                3, 3, subplot_spec=outer[i], wspace=0.05, hspace=0.05
            )
            ax = fig.add_subplot(outer[i])
            ax.axis('off')

            if i >= pms_place.shape[0]:
                continue

            ax.set_title(f"Neuron {unit_names[i]}\nCorr {pm_place_s[i]:.2f}", fontsize=7)

            pm_place = pms_place[i, ...]
            r_angles = np.zeros((n_samples_pos*n_samples_pos))
            r_lens = r_angles.copy()
            for j, (y_bin, x_bin) in enumerate(product(
                range(1, n_samples_pos+1), range(1, n_samples_pos+1))
            ):
                pm = pm_place[x_bin-1, y_bin-1]
                r_len, r_angle = resultant_vector(
                    bins_thet,
                    nanrobust=False,
                    w=pm/pm.sum() if pm.sum()!=0 else pm,
                    d=2*np.pi/n_samples_thet
                )
                r_angles[j] = r_angle*180/np.pi
                r_lens[j] = r_len
                ax = fig.add_subplot(inner[j], polar=True)

                ax.plot(
                    np.append(thetas_ticks, thetas_ticks[0]), # we want to close the circle
                    np.append(pm, pm[0]), # we want to close the circle
                    lw=1.5, c='blue',
                )
                if r_angle:
                    ax.vlines(
                        r_angle,
                        0, np.max(pm),
                        colors='red', lw=1, #linestyles='dashed'
                    )
                ax.set_xticklabels([]) # remove degrees indication
                ax.set_rticks([max(pm)], labels=[f"{max(pm):.2f}"]) # add intensity indication
                ax.tick_params(axis='y', labelsize=5)
                ax.set_theta_direction(-1)
                ax.set_theta_zero_location('N') # move 0 to the north
                ax.grid(True)

        if self.save_figures:
            plt.savefig(os.path.join(self.exp_dir, f'polar_maps_place{appendix}.png'), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def hd_units_with_place(
        self, ncols:int, polar_maps:np.array, rate_maps:np.array,
        selected_place_hd_units:np.array, thetas_ticks:np.array,
    ):
        polar_maps = polar_maps[selected_place_hd_units, ...]
        rate_maps = rate_maps[selected_place_hd_units, ...]

        nrows = int(np.ceil(rate_maps.shape[0]/ncols))

        fig = plt.figure(figsize=(2.5*ncols, 1.5*nrows))
        outer = gridspec.GridSpec(nrows, ncols)

        for i in range(nrows*ncols):
            inner = gridspec.GridSpecFromSubplotSpec(
                1, 2, subplot_spec=outer[i], wspace=0.05, hspace=0.05
            )
            ax = plt.Subplot(fig, outer[i])
            ax.axis('off')

            if i >= len(selected_place_hd_units):
                continue

            ax.set_title(f"Neuron {selected_place_hd_units[i]}\nPlace ∧ HD unit", fontsize=7)
            fig.add_subplot(ax)

            # HD unit
            pm = polar_maps[i]
            ax = fig.add_subplot(inner[0], projection='polar')
            ax.plot(
                np.append(thetas_ticks, thetas_ticks[0]),
                np.append(pm, pm[0]),
                lw=3, c='blue',
                # marker='o', ms=5, mfc='red'
            )
            ax.set_xticklabels([]) # remove degrees indication
            ax.set_rticks([max(pm)], labels=[f"{max(pm):.2f}"]) # add intensity indication
            ax.tick_params(axis='y', labelsize=9)
            ax.set_theta_direction(-1)
            ax.set_theta_zero_location('N') # move 0 to the north
            ax.grid(True)

            # Place cells
            rm = rate_maps[i]
            ax = fig.add_subplot(inner[1])
            ax.set_axis_off()
            ax.imshow(rm, cmap='jet')

        if self.save_figures:
            plt.savefig(os.path.join(self.exp_dir, f'hd_units_with_place.png'), bbox_inches='tight')
            plt.close()
        else:
            plt.show()


    def obs_corr_hd_units_with_place(
        self, ncols:int, polar_maps:np.array, polar_maps_corr:np.array, rate_maps:np.array,
        dis_ratios, selected_place_hd_units:np.array, thetas_ticks:np.array,
    ):
        polar_maps = polar_maps[selected_place_hd_units, ...]
        polar_maps_corr = polar_maps_corr[selected_place_hd_units, ...]
        rate_maps = rate_maps[selected_place_hd_units, ...]
        dis_ratios = dis_ratios[selected_place_hd_units]

        nrows = int(np.ceil(rate_maps.shape[0]/ncols))

        fig = plt.figure(figsize=(3*ncols, 1.5*nrows))
        outer = gridspec.GridSpec(nrows, ncols)

        for i in range(nrows*ncols):
            inner = gridspec.GridSpecFromSubplotSpec(
                1, 3, subplot_spec=outer[i], wspace=0.05, hspace=0.05
            )
            ax = plt.Subplot(fig, outer[i])
            ax.axis('off')

            if i >= len(selected_place_hd_units):
                continue

            ax.set_title(f"Neuron {selected_place_hd_units[i]}\nPlace ∧ HD unit (r{dis_ratios[i]:.2f})", fontsize=7)
            fig.add_subplot(ax)

            # HD unit
            pm = polar_maps[i]
            ax = fig.add_subplot(inner[0], projection='polar')
            ax.plot(
                np.append(thetas_ticks, thetas_ticks[0]),
                np.append(pm, pm[0]),
                lw=3, c='blue',
                # marker='o', ms=5, mfc='red'
            )
            ax.set_xticklabels([]) # remove degrees indication
            ax.set_rticks([max(pm)], labels=[f"{max(pm):.2f}"]) # add intensity indication
            ax.tick_params(axis='y', labelsize=9)
            ax.set_theta_direction(-1)
            ax.set_theta_zero_location('N') # move 0 to the north
            ax.grid(True)

            # HD unit
            pm = polar_maps_corr[i]
            ax = fig.add_subplot(inner[1], projection='polar')
            ax.plot(
                np.append(thetas_ticks, thetas_ticks[0]),
                np.append(pm, pm[0]),
                lw=3, c='blue',
                # marker='o', ms=5, mfc='red'
            )
            ax.set_xticklabels([]) # remove degrees indication
            ax.set_rticks([max(pm)], labels=[f"{max(pm):.2f}"]) # add intensity indication
            ax.tick_params(axis='y', labelsize=9)
            ax.set_theta_direction(-1)
            ax.set_theta_zero_location('N') # move 0 to the north
            ax.grid(True)

            # Place cells
            rm = rate_maps[i]
            ax = fig.add_subplot(inner[2])
            ax.set_axis_off()
            ax.imshow(rm, cmap='jet')

        if self.save_figures:
            plt.savefig(os.path.join(self.exp_dir, f'obs_pred_hd_units_with_place.png'), bbox_inches='tight')
            plt.close()
        else:
            plt.show()


    def tmp(
        self, ncols:int, polar_maps:np.array, polar_maps_corr:np.array, polar_maps_pred:np.array,
        dis_ratios, selected_place_hd_units:np.array, thetas_ticks:np.array,
    ):
        polar_maps = polar_maps[selected_place_hd_units, ...]
        polar_maps_corr = polar_maps_corr[selected_place_hd_units, ...]
        polar_maps_pred = polar_maps_pred[selected_place_hd_units, ...]
        dis_ratios = dis_ratios[selected_place_hd_units]

        nrows = int(np.ceil(polar_maps.shape[0]/ncols))

        fig = plt.figure(figsize=(3*ncols, 1.5*nrows))
        outer = gridspec.GridSpec(nrows, ncols)

        for i in range(nrows*ncols):
            inner = gridspec.GridSpecFromSubplotSpec(
                1, 3, subplot_spec=outer[i], wspace=0.05, hspace=0.05
            )
            ax = plt.Subplot(fig, outer[i])
            ax.axis('off')

            if i >= len(selected_place_hd_units):
                continue

            ax.set_title(f"Neuron {selected_place_hd_units[i]}\n(r{dis_ratios[i]:.2f})", fontsize=7)
            fig.add_subplot(ax)

            # HD unit
            pm = polar_maps[i]
            ax = fig.add_subplot(inner[0], projection='polar')
            ax.plot(
                np.append(thetas_ticks, thetas_ticks[0]),
                np.append(pm, pm[0]),
                lw=3, c='blue',
                # marker='o', ms=5, mfc='red'
            )
            ax.set_xticklabels([]) # remove degrees indication
            ax.set_rticks([max(pm)], labels=[f"{max(pm):.2f}"]) # add intensity indication
            ax.tick_params(axis='y', labelsize=9)
            ax.set_theta_direction(-1)
            ax.set_theta_zero_location('N') # move 0 to the north
            ax.grid(True)

            # HD unit
            pm = polar_maps_corr[i]
            ax = fig.add_subplot(inner[1], projection='polar')
            ax.plot(
                np.append(thetas_ticks, thetas_ticks[0]),
                np.append(pm, pm[0]),
                lw=3, c='blue',
                # marker='o', ms=5, mfc='red'
            )
            ax.set_xticklabels([]) # remove degrees indication
            ax.set_rticks([max(pm)], labels=[f"{max(pm):.2f}"]) # add intensity indication
            ax.tick_params(axis='y', labelsize=9)
            ax.set_theta_direction(-1)
            ax.set_theta_zero_location('N') # move 0 to the north
            ax.grid(True)

            # HD unit
            pm = polar_maps_pred[i]
            ax = fig.add_subplot(inner[2], projection='polar')
            ax.plot(
                np.append(thetas_ticks, thetas_ticks[0]),
                np.append(pm, pm[0]),
                lw=3, c='blue',
                # marker='o', ms=5, mfc='red'
            )
            ax.set_xticklabels([]) # remove degrees indication
            ax.set_rticks([max(pm)], labels=[f"{max(pm):.2f}"]) # add intensity indication
            ax.tick_params(axis='y', labelsize=9)
            ax.set_theta_direction(-1)
            ax.set_theta_zero_location('N') # move 0 to the north
            ax.grid(True)

        if self.save_figures:
            plt.savefig(os.path.join(self.exp_dir, f'obs_pred_hd_units_with_place.png'), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

