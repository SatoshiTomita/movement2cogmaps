import os
import numpy as np
from typing import List
from itertools import product
import wandb

from utils.metrics import get_smooth_polar_map, get_smooth_rate_map, get_spatial_correlation
from utils.metrics import spatial_info, resultant_vector
from utils.spatial_fields import detect_fields

from matplotlib.spines import Spine
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

C_PLACE_CELLS = '#C5268D'
C_HD_CELLS = '#428808'
C_CONJUNCTIVE_CELLS = '#3589EB'

RM_VS_HD_ROTATIONS = 3 # squared
PM_VS_PLACE_BINS = 3 # squared

WANDB_PLOTS_PREFIX = 'spatial_analysis_plots/'
WANDB_METRICS_PREFIX = 'spatial_analysis_metrics/'
WANDB_HIST_PREFIX = 'spatial_analysis_histograms/'

def wandb_log_hist(hist_np, name):
    occ, bin_e = hist_np
    data = [
        [x, y] for (x, y) in
        zip([(a+b)/2 for a, b in zip(bin_e, bin_e[1:])], occ)
    ]
    table = wandb.Table(data=data, columns=[name, "occ"])
    wandb.log({
        WANDB_HIST_PREFIX+name: wandb.plot.line(
            table, name, "occ", title=name, split_table=True
        )
    })


class RateMaps():
    SIGMA_SMOOTHING = .75
    N_SAMPLES_POS = 25
    PLACE_SI_TH = 0.3

    def __init__(
        self, positions:np.array, env_dim:float
    ):
        """
        Args:
        positions (np.array): (samples, 2) Array of positions.
        env_dim (float): Dimension of the environment.
        """
        
        self.positions = positions
        self.sigma = self.SIGMA_SMOOTHING
        self.env_dim = env_dim
        self.bin_size = env_dim/self.N_SAMPLES_POS


    def calculate_rate_maps(self, latent_activity:np.array):
        """
        Args:
        latent_activity (np.array): (samples, n_units)
            Activity rate for each unit over samples.

        Returns:
        rate_maps (np.array): (n_units, n_samples_pos, n_samples_pos)
            Array of rate maps, one for each unit.
        """
        rate_maps, occ_smoothed = get_smooth_rate_map(
            latent_activity,
            self.positions,
            limits = [(0, self.env_dim), (0, self.env_dim)],
            n_bins = self.N_SAMPLES_POS,
            sigma = self.sigma
        )
        rate_maps = np.moveaxis(rate_maps, -1, 0)

        return rate_maps, occ_smoothed

    def calculate_metrics(self, rate_maps:np.array, occ:np.array, norm:str=None):
        """
        Calculate the spatial information for the rate maps.
        
        Args:
        rate_maps (np.array): (n_units, n_samples_pos, n_samples_pos) Array of rate maps.
        occ (np.array): (n_samples_pos, n_samples_pos) Array of occupancy. Defaults to None: uniform.

        Returns:
        si (np.array): (n_units, 1) Array of spatial information.
        """

        n = self.N_SAMPLES_POS
        if norm is None:
            rate_maps_norm = rate_maps
        elif norm == 'minmax':
            rate_maps_min = np.moveaxis(
                np.tile(np.nanmin(rate_maps, axis=(1,2)), (n, n, 1)), -1, 0
            )
            rate_maps_max = np.moveaxis(
                np.tile(np.nanmax(rate_maps, axis=(1,2)), (n, n, 1)), -1, 0
            )
            rate_maps_norm = np.divide(
                (rate_maps - rate_maps_min),
                (rate_maps_max - rate_maps_min),
                where=np.logical_and(
                    (rate_maps_max - rate_maps_min)!=0,
                    ~np.isnan(rate_maps)
                ),
                out=rate_maps
            )
        elif norm == 'sum':
            _sum = np.moveaxis(
                np.tile(np.sum(rate_maps, axis=(1,2)), (n, n, 1)), -1, 0
            )
            rate_maps_norm = np.divide(
                rate_maps, _sum,
                where=np.logical_and(
                    _sum!=0,
                    ~np.isnan(_sum)
                ),
                out=rate_maps
            )

        si = [spatial_info(rm, occ) for rm in rate_maps_norm]

        return np.array(si)

    def get_place_cells_indices(self, rate_maps:np.array, si:np.array):
        """
        Get the indices of the place cells based on the spatial information.

        Args:
            rate_maps (np.array): (n_units, n_samples_pos, n_samples_pos) Array of rate maps.
            si (np.array): (n_units, 1) Array of spatial information.
        Returns:
            (np.array): Indices of the place cells.
        """

        indices_place_cells = np.array([
            idx for idx in range(rate_maps.shape[0]) if
            (not np.isnan(si[idx])) and (si[idx] > self.PLACE_SI_TH)
        ], dtype=int)

        return indices_place_cells
    
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

        half_idx = int(np.ceil(self.N_SAMPLES_POS/2))

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
        half_idx = int(np.ceil(self.N_SAMPLES_POS/2))

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

    def calculate_rm_vs_hd(self, latent_activity:np.array, thetas:np.array):
        """
        Calculate the rate maps for each head direction.

        Args:
            latent_activity (np.array): (samples, n_units) Activity rate for each unit over samples.
            thetas (np.array): (samples, 1) Head directions for each sample.
        """
        rm_vs_hd = []

        bins_thet = np.linspace(-np.pi, np.pi, RM_VS_HD_ROTATIONS**2)
        indices = np.digitize(thetas, bins_thet)

        for idx in range(1, RM_VS_HD_ROTATIONS**2):
            rm_vs_hd.append(
                get_smooth_rate_map(
                    latent_activity[indices==idx],
                    self.positions[indices==idx],
                    limits = [(0, self.env_dim), (0, self.env_dim)],
                    n_bins = self.N_SAMPLES_POS,
                    sigma = self.sigma
                )[0]
            )
            
        rm_vs_hd = np.stack(rm_vs_hd, axis=0)
        rm_vs_hd = np.moveaxis(rm_vs_hd, -1, 0)
        
        return rm_vs_hd

    def rm_vs_hd_stability(self, rate_maps:np.array, rm_vs_hd:np.array):
        """
        Comparison between two sets of place cells.

        Args:
            rate_maps (np.array): (n_units, n_samples_pos, n_samples_pos) Array of neurons' activities.
            rm_vs_hd (np.array): (n_units, n_rotations, n_samples_pos, n_samples_pos) Array of neurons' activities for each head direction.
        
        Returns:
            (np.array): (n_units, 1) The correlation for each rate map.
        """
        correlations = np.zeros(rate_maps.shape[0])
        for h in range(rm_vs_hd.shape[1]):
            rm_hd = rm_vs_hd[:, h, ...]
            
            corr = [
                get_spatial_correlation(rate_maps[i], rm_hd[i])[0]\
                for i in range(rate_maps.shape[0])
            ]
            correlations += corr

        correlations /= rm_vs_hd.shape[1]
        return correlations


class RateMapsPlotter():

    def __init__(self, exp_dir:str, save_figures:bool=True, wandb_log:bool=False):
        """
        Args:
        exp_dir (str): Path to the experiment directory.
        save_figures (bool): Flag to indicate if the figures should be saved. Defaults to True.
        """
        
        self.exp_dir = exp_dir
        self.save_figures = save_figures
        self.wandb_log = wandb_log

    def average_rate_map(self, rate_maps:np.array):
        """
        Plot the average rate map.
        
        Args:
        rate_maps (np.array): (n_units, n_samples_pos, n_samples_pos) Array of rate maps.
        """
        fig = plt.figure(figsize=(4,4))
        plt.imshow(np.mean(rate_maps, axis=0), cmap='jet')
        plt.colorbar()
        plt.axis('off')
        plt.tight_layout()
        if self.wandb_log:
            wandb.log({WANDB_PLOTS_PREFIX+'avg_rate_map' : wandb.Image(fig)})
        if self.save_figures:
            plt.savefig(os.path.join(self.exp_dir, 'average_rate_map.png'), bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def metric_histogram(self, metric:np.array, metric_name:str):
        """
        Plot a histogram of the given metric.
        
        Args:
        metric (np.array): (n_units, 1) Array of the metric to plot.
        metric_name (str): Name of the metric to plot.
        """
        plt.figure(figsize=(5,4))
        plt.hist(metric, bins=50)
        plt.xlabel(metric_name)
        plt.ylabel('Occurrences')
        if 'sir' in metric_name.lower():
            lims = (0, 1.5)
            plt.xlim(*lims)
        plt.tight_layout()

        if self.wandb_log:
            metric_hist_np = np.histogram(metric, bins=50, range=lims)
            wandb_log_hist(metric_hist_np, metric_name)
            wandb.log({WANDB_METRICS_PREFIX+metric_name: float(np.nanmean(metric))})

        if self.save_figures:
            plt.savefig(os.path.join(self.exp_dir, f'{metric_name}.png'), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def rate_maps(
        self, ncols:int, rate_maps:np.array, si:np.array,
        indices_place_cells:np.array=np.array([],dtype=int),
        indices_hd_cells:np.array=np.array([],dtype=int),
        indices_conjunctive_cells:np.array=np.array([],dtype=int),
        appendix:str=''
    ):
        """
        Plot the rate maps.
        
        Args:
        ncols (int): Number of columns for the plot.
        rate_maps (np.array): (n_units, n_samples_pos, n_samples_pos) Array of rate maps.
        si (np.array): (n_units, 1) Array of spatial information.
        indices_place_cells (List[int], optional): Indices of units classified as place cells. Defaults to [], no highlighting.
        indices_hd_cells (List[int], optional): Indices of units classified as head direction cells. Defaults to [], no highlighting.
        indices_conjunctive_cells (List[int], optional): Indices of units classified as conjunctive cells. Defaults to [], no highlighting.
        appendix (str, optional): Appendix for the file name. Defaults to ''.
        """
        nrows = int(np.ceil(rate_maps.shape[0]/ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(8.27, 11.69*4), dpi=300)

        indices = np.concat((
            indices_place_cells,
            indices_conjunctive_cells,
            indices_hd_cells,
            np.setdiff1d(
                np.arange(rate_maps.shape[0], dtype=int),
                np.concat((indices_place_cells, indices_conjunctive_cells, indices_hd_cells))
            )
        ))

        for j in range(nrows*ncols):
            ax = axs.flat[j]
            if j >= len(indices):
                ax.axis('off')
                continue
            idx = indices[j]

            if idx in indices_place_cells:
                color = C_PLACE_CELLS
            elif idx in indices_conjunctive_cells:
                color = C_CONJUNCTIVE_CELLS
            elif idx in indices_hd_cells:
                color = C_HD_CELLS
            else : color = None

            if color is not None:
                for child in ax.get_children():
                    if isinstance(child, Spine):
                        child.set_color(color)
                        child.set_linewidth(3)
            else:
                ax.axis('off')

            ax.set_xticks([])
            ax.set_yticks([])
            
            rm = rate_maps[idx, ...]

            ax.imshow(rm, cmap='jet')
            ax.set_title(
                f'{idx:03d}: [{rm.min():.2f},{rm.max():.2f}]\n'+\
                f'SI {si[idx]:.2f}', fontsize=4.25
            )

        plt.tight_layout(w_pad=2.0, h_pad=1.5)
        if self.wandb_log:
            wandb.log({WANDB_PLOTS_PREFIX+'rate_maps' : wandb.Image(fig)})
        if self.save_figures:
            plt.savefig(os.path.join(self.exp_dir, f'rate_maps{appendix}.png'))
            plt.close()
        else:
            plt.show()

    def rate_maps_vs_hd(
        self, ncols:int, rate_maps:np.array, rm_vs_hd:np.array,
        rm_vs_hd_stability:np.array, frac:float,
        indices_place_cells:np.array=np.array([],dtype=int),
        indices_hd_cells:np.array=np.array([],dtype=int),
        indices_conjunctive_cells:np.array=np.array([],dtype=int),
        appendix:str=''
    ):
        """
        Plot the rate maps for each head direction.
        
        Args:
            ncols (int): Number of columns for the plot.
            rate_maps (np.array): (n_units, n_samples_pos, n_samples_pos) Array of rate maps.
            rm_vs_hd (np.array): (n_units, n_rotations, n_samples_pos, n_samples_pos) Array of rate maps for each head direction.
            rm_vs_hd_stability (np.array): (n_units, 1) Array of correlations for each rate map.
            frac (float, optional): Fraction of neurons to plot.
            indices_place_cells (List[int], optional): Indices of units classified as place cells. Defaults to [], no highlighting.
            indices_hd_cells (List[int], optional): Indices of units classified as hd cells. Defaults to [], no highlighting.
            indices_conjunctive_cells (List[int], optional): Indices of units classified as conjunctive cells. Defaults to [], no highlighting.
            appendix (str, optional): Appendix for the file name. Defaults to ''.
        """
        random_indices = []
        for indices in [
            indices_place_cells, indices_conjunctive_cells, indices_hd_cells, 
            np.setdiff1d(
                np.arange(rate_maps.shape[0]),
                np.concat((indices_place_cells, indices_hd_cells, indices_conjunctive_cells))
            )
        ]:
            random_indices += list(np.sort(np.random.choice(
                indices, size=int(len(indices)*frac), replace=False
            )))

        nrows = int(np.ceil(len(random_indices)/ncols))

        fig = plt.figure(figsize=(8.27, 11.69+frac*11.69*8), dpi=300)
        outer = gridspec.GridSpec(nrows, ncols)

        for j in range(nrows*ncols):
            inner = gridspec.GridSpecFromSubplotSpec(
                RM_VS_HD_ROTATIONS, RM_VS_HD_ROTATIONS,
                subplot_spec=outer[j], wspace=0.05, hspace=0.05
            )
            ax = fig.add_subplot(outer[j])
            ax.axis('off')

            if j >= len(random_indices):
                continue
            idx = random_indices[j]

            if idx in indices_place_cells:
                color = C_PLACE_CELLS
            elif idx in indices_conjunctive_cells:
                color = C_CONJUNCTIVE_CELLS
            elif idx in indices_hd_cells:
                color = C_HD_CELLS
            else : color = None
                
            ax.set_title(
                f"{idx:03d}: Corr {rm_vs_hd_stability[idx]:.2f}",
                fontsize=5
            )

            rm_hd = rm_vs_hd[idx, ...]

            rot_idx_order = [3, 4, 5, 2, 6, 1, 0, 7]
            delta = 0
            for j in range(RM_VS_HD_ROTATIONS**2):
                ax = fig.add_subplot(inner[j])
                if color is not None:
                    for child in ax.get_children():
                        if isinstance(child, Spine):
                            child.set_color(color)
                            child.set_linewidth(2)
                else:
                    ax.axis('off')
                ax.set_xticks([])
                ax.set_yticks([])

                if j==4:
                    ax.imshow(rate_maps[idx, ...], cmap='jet')
                    delta = -1
                    continue

                ax.imshow(rm_hd[rot_idx_order[j+delta]], cmap='jet')

        plt.tight_layout(w_pad=2.0, h_pad=0.5)
        if self.wandb_log:
            wandb.log({WANDB_PLOTS_PREFIX+'rate_maps_vs_hd' : wandb.Image(fig)})
        if self.save_figures:
            fig.savefig(os.path.join(self.exp_dir, f'rm_vs_hd{appendix}.png'), bbox_inches='tight')
            plt.close()
        else:
            plt.show()




class PolarMaps():
    SIGMA_SMOOTHING = .75
    N_SAMPLES_THET = 60
    HD_SI_TH = 0.2
    HD_RVL_TH = 0.3

    MIN_MAX_SCALING = False
    
    def __init__(
        self, thetas:np.array
    ):
        """
        Args:
            thetas (np.array): (samples, 1) Array of head directions.
        """
        self.thetas = thetas

        # create directionality bins and indices from -pi to pi
        bins_thet = self.get_bins_thet()
        self.indices = np.digitize(self.thetas, bins_thet) if self.thetas is not None else None
        # the tick sits in the middle of the bins
        self.thetas_ticks = self.get_thetas_ticks()
        self.bins_thet = bins_thet[:-1]

    @staticmethod
    def get_bins_thet():
        return np.linspace(-np.pi, np.pi, PolarMaps.N_SAMPLES_THET+1)
    
    @staticmethod
    def get_thetas_ticks():
        bins_thet = PolarMaps.get_bins_thet()
        return np.array(
            [np.mean([a, b]) for a, b in zip(bins_thet, bins_thet[1:])]
        )

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
        polar_maps, occ_smoothed = get_smooth_polar_map(
            latent_activity,
            self.thetas,
            n_bins=self.N_SAMPLES_THET,
            sigma=5
        )
        # pm_std = [
        #     np.std(latent_activity[self.indices==idx], axis=0)\
        #     for idx in range(1, self.n_samples_thet+1)
        # ]
        return polar_maps, occ_smoothed
    
    def calculate_metrics(self, polar_maps:np.array, occ:np.array):
        """
        Calculate the spatial information, Rayleigh Vector length and KL divergence for the polar maps.
        
        Args:
        polar_maps (np.array): (n_units, n_samples_thet) Array of polar maps.
        occ (np.array): (n_samples_thet) Array of occupancies.
        
        Returns:
        si (np.array): (n_units, 1) Array of spatial information.
        rvl (np.array): (n_units, 1) Array of RV lengths.
        rvangle (np.array): (n_units, 1) Array of RV angles.
        """
        n = polar_maps.shape[-1]

        divisor = np.moveaxis(
            np.tile(polar_maps.sum(axis=1), (n, 1)), -1, 0
        )
        polar_maps_norm = np.divide(
            polar_maps, divisor,
            where=divisor != 0, out=polar_maps
        )

        si = [spatial_info(pm, occ) for pm in polar_maps_norm]
        rv = [
            resultant_vector(
                self.bins_thet,
                nanrobust=False,
                w=pm,
                d=2*np.pi/self.N_SAMPLES_THET
            ) for pm in polar_maps_norm
        ]

        rvl = [x[0] for x in rv]
        rvangle = [x[1] for x in rv]

        return np.array(si), np.array(rvl), np.array(rvangle)

    def get_hd_cells_indices(self, polar_maps:np.array, si:np.array, rvl:np.array):
        """
        Get the indices of the head direction cells based on the spatial information and Rayleigh Vector length.
        Args:
            polar_maps (np.array): (n_units, n_samples_thet) Array of polar maps.
            si (np.array): (n_units, 1) Array of spatial information.
            rvl (np.array): (n_units, 1) Array of Rayleigh Vector lengths.
        Returns:
            (np.array): Indices of the head direction cells.
        """
        indices_hd_cells = np.array([
            idx for idx in range(polar_maps.shape[0]) if
            (si[idx] > self.HD_SI_TH) or (rvl[idx] > self.HD_RVL_TH)
        ], dtype=int)

        return indices_hd_cells

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
      
    def calculate_pm_vs_place(
        self, latent_activity:np.array, positions:np.array, env_dim:float
    ):
        """
        Calculate the polar data for the given latent data subdivided by positional bins.

        Args:
            latent_activity (np.array): (n_samples, n_units) The latent data, hidden units' activity.
            positions (np.array): (n_samples, 2) Array of positions.
            env_dim (float): Dimension of the environment.

        Returns:
            np.array: (n_neurons, n_bins, n_samples_pos, n_samples_thet) The latent activity in a polar space and in positional bins.
        """
        n_bins = PM_VS_PLACE_BINS
        indices_pos = np.digitize(positions, bins=np.linspace(0, env_dim, n_bins+1))

        polar_maps_place = np.zeros(
            (n_bins, n_bins, latent_activity.shape[-1], self.N_SAMPLES_THET)
        )

        for x_bin, y_bin in product(range(1, n_bins+1), range(1, n_bins+1)):
            idx_pos = np.logical_and(indices_pos[:,0]==x_bin, indices_pos[:,1]==y_bin)
            polar_maps_place[x_bin-1, y_bin-1, ...], _ =\
                get_smooth_polar_map(
                    latent_activity[idx_pos],
                    self.thetas[idx_pos],
                    n_bins=self.N_SAMPLES_THET,
                    sigma=5
                )
        polar_maps_place = np.moveaxis(polar_maps_place, -2, 0)

        return polar_maps_place

    def pm_vs_place_stability(self, polar_maps:np.array, pm_vs_place:np.array):
        """
        Comparison between two sets of place cells.

        Args:
            polar_maps (np.array): (n_units, n_samples_thet) Array of neurons' activities.
            pm_vs_place (np.array): (n_units, n_samples_pos, n_samples_pos, n_samples_thet) Array of neurons' activities in positional bins.

        Returns:
            (np.array): (n_units, 1) The correlation for each polar map.
        """
        correlations = np.zeros(polar_maps.shape[0])

        for p_row in range(pm_vs_place.shape[1]):
            for p_col in range(pm_vs_place.shape[1]):
                pm_place = pm_vs_place[:, p_row, p_col]
                
                corr = [
                    get_spatial_correlation(polar_maps[i], pm_place[i])[0]\
                    for i in range(polar_maps.shape[0])
                ]
                correlations += corr

        correlations /= (pm_vs_place.shape[1]**2)
        return correlations


class PolarMapsPlotter():

    def __init__(self, exp_dir:str, thetas_ticks:np.array, save_figures:bool=True, wandb_log:bool=False):
        """
        Args:
            exp_dir (str): Path to the experiment directory.
            thetas_ticks (np.array): (n_samples_thet) Array of ticks for the polar maps.
        """

        self.exp_dir = exp_dir
        self.tts = thetas_ticks

        self.save_figures = save_figures
        self.wandb_log = wandb_log

    def single_polar_map(self, ax, pm):
        ax.plot(
            np.append(self.tts, self.tts[0]), # we want to close the circle
            np.append(pm, pm[0]), # we want to close the circle
            c='darkgreen', zorder=5, lw=2
            # marker='o', ms=5, mfc='red'
        )
        ax.set_xticklabels([]) # remove degrees indication
        ax.set_rticks([]) # remove intensity indication
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N') # move 0 to the north

        max_radius = np.max(pm)
        padding = max_radius * 0.15
        ax.set_rmax(max_radius + padding)
        ax.set_thetagrids([0, 90, 180, 270])
        ax.grid(True)

    def average_polar_map(self, polar_maps:np.array):
        """
        Plot the average polar map.
        
        Args:
            polar_maps (np.array): (n_units, n_samples_thet) Array of polar maps.
        """
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        polar_map = np.mean(polar_maps, axis=0)
        self.single_polar_map(ax, polar_map)
        
        ax.set_title('Average polar map')
        plt.tight_layout()
        if self.wandb_log:
            wandb.log({WANDB_PLOTS_PREFIX+'avg_polar_map' : wandb.Image(fig)})
        if self.save_figures:
            plt.savefig(os.path.join(self.exp_dir, 'average_polar_map.png'), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def metric_histogram(self, metric:np.array, metric_name:str):
        """
        Plot a histogram of the given metric.
        
        Args:
        metric (np.array): (n_units, 1) Array of the metric to plot.
        metric_name (str): Name of the metric to plot.
        """
        fig = plt.figure(figsize=(5,4))
        plt.hist(metric, bins=50)
        plt.xlabel(metric_name)
        plt.ylabel('Occurrences')
        if 'sid' in metric_name.lower():
            lims = (0, 0.5)
            plt.xlim(*lims)
        elif 'rvl' in metric_name.lower():
            lims = (0, 0.6)
            plt.xlim(*lims)
        plt.tight_layout()

        if self.wandb_log:
            metric_hist_np = np.histogram(metric, bins=50, range=lims)
            wandb_log_hist(metric_hist_np, metric_name)
            wandb.log({WANDB_METRICS_PREFIX+metric_name: float(np.nanmean(metric))})

        if self.save_figures:
            plt.savefig(os.path.join(self.exp_dir, f'{metric_name}.png'), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def polar_maps(
        self, ncols:int, polar_maps:np.array,
        si:np.array, rvl:np.array, rvangle:np.array=None,
        indices_place_cells:np.array=np.array([],dtype=int),
        indices_hd_cells:np.array=np.array([],dtype=int),
        indices_conjunctive_cells:np.array=np.array([],dtype=int),
        appendix:str=''
    ):
        """
        Plot the polar maps.
        
        Args:
            ncols (int): Number of columns for the plot.
            polar_maps (np.array): (n_units, n_samples_thet) Array of polar maps.
            si (np.array): (n_units, 1) Array of Spatial information.
            rvl (np.array): (n_units, 1) Array of RV lengths.
            rvangle (np.array): (n_units, 1) Array of RV angles.
            indices_place_cells (List[int], optional): Indices of units classified as place cells. Defaults to [], no highlighting.
            indices_hd_cells (List[int], optional): Indices of units classified as head direction cells. Defaults to [], no highlighting.
            indices_conjunctive_cells (List[int], optional): Indices of units classified as conjunctive cells. Defaults to [], no highlighting.
            appendix (str, optional): Appendix for the file name. Defaults to ''.
        """
        nrows = int(np.ceil(polar_maps.shape[0]/ncols))
        fig, axs = plt.subplots(
            nrows, ncols, subplot_kw={'projection': 'polar'},
            figsize=(8.27, 11.69*4), dpi=300
        )
        
        indices = np.concat((
            indices_hd_cells,
            indices_conjunctive_cells,
            indices_place_cells,
            np.setdiff1d(
                np.arange(polar_maps.shape[0]),
                np.concat((indices_place_cells, indices_conjunctive_cells, indices_hd_cells))
            )
        ), dtype=int)

        for j in range(nrows*ncols):
            ax = axs.flat[j]
            if j >= len(indices):
                ax.axis('off')
                continue
            idx = indices[j]

            if idx in indices_hd_cells:
                color = C_HD_CELLS
            elif idx in indices_conjunctive_cells:
                color = C_CONJUNCTIVE_CELLS
            elif idx in indices_place_cells:
                color = C_PLACE_CELLS
            else : color = None

            polar_map = polar_maps[idx, ...]
            if np.isnan(polar_map).all(): print("polar map is completely nan")

            self.single_polar_map(ax, polar_map)
            if color is not None:
                ax.set_facecolor((color, 0.2))
            
            ax.spines[['polar']].set_visible(False)

            if rvangle is not None:
                ax.vlines(
                    rvangle[idx], 0, np.max(polar_map),
                    colors='gray', lw=2
                )
            ax.set_title(
                f'{idx:03d} [{min(polar_map):.2f}, {max(polar_map):.2f}]\n'+\
                f'SI {si[idx]:.2f} RVL {rvl[idx]:.2f}', fontsize=4.25
            )

        plt.tight_layout(w_pad=2.0, h_pad=1.5)
        if self.wandb_log:
            wandb.log({WANDB_PLOTS_PREFIX+'polar_maps' : wandb.Image(fig)})
        if self.save_figures:
            plt.savefig(os.path.join(self.exp_dir, f'polar_maps{appendix}.png'), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def polar_maps_vs_place(
        self, ncols:int, pm_vs_place:np.array,
        pm_vs_place_stability:np.array, frac:float,
        indices_place_cells:np.array=np.array([],dtype=int),
        indices_hd_cells:np.array=np.array([],dtype=int),
        indices_conjunctive_cells:np.array=np.array([],dtype=int),
        appendix:str=''
    ):
        """
        Plot the polar maps subdivided by positional bins.
        
        Args:
            ncols (int): Number of columns for the plot.
            pm_vs_place (np.array): (n_units, n_samples_pos, n_samples_pos, n_samples_thet) Array of polar maps in positional bins.
            pm_vs_place_stability (np.array): (n_units, 1) Array of correlations for each polar map.
            frac (float, optional): Fraction of neurons to plot.
            indices_place_cells (List[int], optional): Indices of units classified as place cells. Defaults to [], no highlighting.
            indices_hd_cells (List[int], optional): Indices of units classified as hd cells. Defaults to [], no highlighting.
            indices_conjunctive_cells (List[int], optional): Indices of units classified as conjunctive cells. Defaults to [], no highlighting.
            appendix (str, optional): Appendix for the file name. Defaults to ''.
        """
        random_indices = []
        for indices in [
            indices_hd_cells, indices_conjunctive_cells, indices_place_cells,
            np.setdiff1d(
                np.arange(pm_vs_place.shape[0]),
                np.concat((indices_place_cells, indices_hd_cells, indices_conjunctive_cells)))
        ]:
            random_indices += list(np.sort(np.random.choice(
                indices, size=int(len(indices)*frac), replace=False
            )))

        nrows = int(np.ceil(len(random_indices)/ncols))

        fig = plt.figure(figsize=(8.27, 11.69+frac*11.69*8), dpi=300)
        outer = gridspec.GridSpec(nrows, ncols)

        for j in range(nrows*ncols):
            inner = gridspec.GridSpecFromSubplotSpec(
                PM_VS_PLACE_BINS, PM_VS_PLACE_BINS,
                subplot_spec=outer[j], wspace=0.05, hspace=0.05
            )
            ax = fig.add_subplot(outer[j])
            ax.axis('off')
            if j >= len(random_indices):
                continue
            idx = random_indices[j]

            if idx in indices_hd_cells:
                color = C_HD_CELLS
            elif idx in indices_conjunctive_cells:
                color = C_CONJUNCTIVE_CELLS
            elif idx in indices_place_cells:
                color = C_PLACE_CELLS
            else : color = None

            ax.set_title(
                f'{idx:03d}: Corr {pm_vs_place_stability[idx]:.2f}',
                fontsize=6
            )

            pm_place = pm_vs_place[idx, ...]
            for j, (y_bin, x_bin) in enumerate(product(
                range(1, PM_VS_PLACE_BINS+1), range(1, PM_VS_PLACE_BINS+1))
            ):
                pm = pm_place[x_bin-1, y_bin-1]
                ax = fig.add_subplot(inner[j], polar=True)

                if color is not None:
                    ax.set_facecolor((color, 0.2))
                
                ax.spines[['polar']].set_visible(False)

                self.single_polar_map(ax, pm)

        plt.tight_layout(w_pad=2.0, h_pad=0.5)
        if self.wandb_log:
            wandb.log({WANDB_PLOTS_PREFIX+'polar_maps_vs_place' : wandb.Image(fig)})
        if self.save_figures:
            plt.savefig(os.path.join(self.exp_dir, f'pm_vs_place{appendix}.png'), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def conjunctive_rms_pms(
        self, ncols:int, rate_maps:np.array, polar_maps:np.array,
        indices_conjunctive_cells:np.array,
    ):
        polar_maps = polar_maps[indices_conjunctive_cells, ...]
        rate_maps = rate_maps[indices_conjunctive_cells, ...]

        nrows = int(np.ceil(rate_maps.shape[0]/ncols))

        fig = plt.figure(figsize=(8.27, 11.69+len(indices_conjunctive_cells)/500*11.69*8), dpi=300)
        outer = gridspec.GridSpec(nrows, ncols)

        for idx in range(nrows*ncols):
            inner = gridspec.GridSpecFromSubplotSpec(
                1, 2, subplot_spec=outer[idx], wspace=0.05, hspace=0.05
            )
            ax = fig.add_subplot(outer[idx])
            ax.axis('off')

            if idx >= len(indices_conjunctive_cells):
                continue

            ax.set_title(f"{indices_conjunctive_cells[idx]:03d}", fontsize=6)
            fig.add_subplot(ax)

            # polar map
            pm = polar_maps[idx]
            ax = fig.add_subplot(inner[0], projection='polar')
            self.single_polar_map(ax, pm)

            # rate map
            rm = rate_maps[idx]
            ax = fig.add_subplot(inner[1])
            ax.set_axis_off()
            ax.imshow(rm, cmap='jet')

        plt.tight_layout(w_pad=2.0, h_pad=1.5)
        if self.wandb_log:
            wandb.log({WANDB_PLOTS_PREFIX+'conjunctive_rms_pms' : wandb.Image(fig)})
        if self.save_figures:
            plt.savefig(os.path.join(self.exp_dir, f'conjunctive_rms_pms.png'), bbox_inches='tight')
            plt.close()
        else:
            plt.show()
