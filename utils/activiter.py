import torch
import numpy as np
import os
import json
import wandb

from utils.spatial_units import (
    RateMaps, RateMapsPlotter, PolarMaps, PolarMapsPlotter, wandb_log_hist, WANDB_METRICS_PREFIX
)
from utils.plots import plot_trajectory_heatmap


class RNNActiviter():

    def __init__(self, args, data_dir, device, model_name, exp_dir):
        self.args = args
        self.data_dir = data_dir
        self.device = device
        self.model_name = model_name
        self.exp_dir = exp_dir

        self.init_default_args()

    def init_default_args(self):
        if self.args.behaviour_act is None:
            self.args.behaviour_act = self.args.behaviour

    def redefine_exp_dir(self):
        dir_name = f"act_{self.args.behaviour_act}_epoch{self.args.epoch_act}"
        self.exp_dir = os.path.join(self.exp_dir, dir_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        print(f"\n[+] Created activity directory\n\t{self.exp_dir}")

    def load_model(self):
        import re

        if self.args.epoch_act is not None:
            epoch = self.args.epoch_act
        else:
            rnn_files = [f for f in os.listdir(self.exp_dir) if re.match(r"rnn_epoch\d+\.pth", f)]
            epoch = max([int(re.search(r'\d+', f).group()) for f in rnn_files])
        
        self.args.epoch_act = epoch
        load_model_dir = os.path.join(self.exp_dir, f"rnn_epoch{epoch}.pth")
        print(f"\n[*] Loading model from\n\t{load_model_dir}")
        if self.device != 'cuda':
            rnn = torch.load(
                load_model_dir,
                weights_only=False,
                map_location=torch.device(self.device)
            ).to(self.device)

        self.redefine_exp_dir()

        return rnn, epoch, self.exp_dir
    
    def extract_latent_activity(
        self, rnn, dataloader, trainer_bptt, save_output=True
    ):
        self.args.clip_value = None

        vloss_dict, latent_activity, positions, thetas, _, _, _ =\
            trainer_bptt.test_epoch(rnn, dataloader, for_trajectory=True)

        for k, v in vloss_dict.items():
            vloss_dict[k] = float(v)
            print(f"\t{k}: {v:.5f}")

        latent_activity = np.concatenate(latent_activity, axis=1)
        positions = np.concatenate(positions, axis=1)
        thetas = np.concatenate(thetas, axis=1)

        if save_output:
            np.save(os.path.join(self.exp_dir, 'latent_activity.npy'), latent_activity)
            np.save(os.path.join(self.exp_dir, 'positions.npy'), positions)
            np.save(os.path.join(self.exp_dir, 'thetas.npy'), thetas)
            with open(os.path.join(self.exp_dir, 'vloss_dict.json'), 'w') as f:
                json.dump(vloss_dict, f, indent=4)

        return latent_activity, positions, thetas, vloss_dict
    
    def split_data(self, latent_activity, positions, thetas):
        # generate a meaningful split to calculate the cells stability later

        # if there are more than 1 seed for validation, split experiments by seed
        if positions.shape[0] > len(self.args.seeds_act):
            idx_half = latent_activity.shape[0]//2
            latent_activity_half1 = latent_activity[:idx_half].reshape(-1, latent_activity.shape[-1])
            latent_activity_half2 = latent_activity[idx_half:].reshape(-1, latent_activity.shape[-1])
            positions_half1 = positions[:idx_half].reshape(-1, positions.shape[-1])
            positions_half2 = positions[idx_half:].reshape(-1, positions.shape[-1])
            thetas_half1 = thetas[:idx_half]
            if len(thetas_half1.shape) > 1 : thetas_half1 = thetas_half1.reshape(-1, thetas.shape[-1]).squeeze()
            thetas_half2 = thetas[idx_half:]
            if len(thetas_half2.shape) > 1 : thetas_half2 = thetas_half2.reshape(-1, thetas.shape[-1]).squeeze()
        # if there is only one seed for validation, split experiments by time
        else:
            idx_half = latent_activity.shape[1]//2
            latent_activity_half1 = latent_activity[:, :idx_half].reshape(-1, latent_activity.shape[-1])
            latent_activity_half2 = latent_activity[:, idx_half:].reshape(-1, latent_activity.shape[-1])
            positions_half1 = positions[:, :idx_half].reshape(-1, positions.shape[-1])
            positions_half2 = positions[:, idx_half:].reshape(-1, positions.shape[-1])
            thetas_half1 = thetas[:, :idx_half].reshape(-1, thetas.shape[-1]).squeeze()
            thetas_half2 = thetas[:, idx_half:].reshape(-1, thetas.shape[-1]).squeeze()

        print(f"\tHalves shapes: {latent_activity_half1.shape}, {latent_activity_half2.shape}")
        return latent_activity_half1, positions_half1, thetas_half1,\
            latent_activity_half2, positions_half2, thetas_half2

    def trajectory_heatmap(self, positions):
        plot_trajectory_heatmap(self.exp_dir, positions, n_bins=25)
        

    def rnn_place_activity(
        self,
        latent_activity, lact_half1, lact_half2,
        positions, pos_half1, pos_half2,
        thetas
    ):
        exp_dir_place = os.path.join(self.exp_dir, 'place')
        os.makedirs(exp_dir_place, exist_ok=True)

        rm_helper = RateMaps(positions, self.args.env_dim)
        rate_maps, occ = rm_helper.calculate_rate_maps(latent_activity)
        np.save(os.path.join(exp_dir_place, 'rate_maps.npy'), rate_maps)

        print("\tCalculating metric SIr")
        si_r = rm_helper.calculate_metrics(rate_maps.copy(), occ, norm=self.args.ratemap_norm)
        np.save(os.path.join(exp_dir_place, 'si.npy'), si_r)

        rm_stability = None
        rm_half1, rm_half2 = None, None
        try:
            print("\tRate maps stability")
            pu_half1 = RateMaps(pos_half1, self.args.env_dim)
            rm_half1, _ = pu_half1.calculate_rate_maps(lact_half1)

            pu_half2 = RateMaps(pos_half2, self.args.env_dim)
            rm_half2, _ = pu_half2.calculate_rate_maps(lact_half2)

            rm_stability = rm_helper.rate_maps_stability(rm_half1, rm_half2)
            np.save(os.path.join(exp_dir_place, 'rm_half1.npy'), rm_half1)
            np.save(os.path.join(exp_dir_place, 'rm_half2.npy'), rm_half2)
            np.save(os.path.join(exp_dir_place, 'rm_stability.npy'), rm_stability)
        except Exception as e:
            print("\t[-] Rate maps calculation for half of the trajectory failed because\n")
            print(f"\t{e}\n")
            pass

        print("\tRate maps field detection")
        n_fields, rm_fields = rm_helper.rate_maps_field_detection(rate_maps, rm_half1, rm_half2)
        np.save(os.path.join(exp_dir_place, 'n_fields.npy'), n_fields)

        single_field_dim = []
        for fields in rm_fields:
            if fields:
                single_field_dim.append(int(np.mean(np.nansum(np.array(fields)>0, axis=(1, 2)))))
            else: single_field_dim.append(0)
        single_field_dim = np.array(single_field_dim)
        np.save(os.path.join(exp_dir_place, 'single_field_dim.npy'), single_field_dim)

        print("\tAverage rate maps flipped (for border preference)")
        rm_flipped = rm_helper.rm_flipped(rate_maps)
        np.save(os.path.join(exp_dir_place, 'rm_flipped.npy'), rm_flipped)

        print("\tCalculating rate maps vs HD")
        rm_vs_hd = rm_helper.calculate_rm_vs_hd(latent_activity, thetas)
        rm_vs_hd_stability = rm_helper.rm_vs_hd_stability(rate_maps, rm_vs_hd)
        np.save(os.path.join(exp_dir_place, 'rm_vs_hd.npy'), rm_vs_hd)
        np.save(os.path.join(exp_dir_place, 'rm_vs_hd_stability.npy'), rm_vs_hd_stability)

        indices_place_cells = rm_helper.get_place_cells_indices(rate_maps, si_r)

        if len(indices_place_cells) > 0 and self.args.wandb:
            wandb.log({
                WANDB_METRICS_PREFIX+'place cells N fields': float(np.nanmean(n_fields[indices_place_cells])),
                WANDB_METRICS_PREFIX+'place cells field dim': float(np.nanmean(single_field_dim[indices_place_cells])),
                WANDB_METRICS_PREFIX+'place cells stability': float(np.nanmean(rm_stability[indices_place_cells])),
            })

        return rate_maps, si_r, indices_place_cells, n_fields, rm_stability, single_field_dim, rm_vs_hd, rm_vs_hd_stability
    
    def save_place_plots(self, rate_maps, si_r, indices_place_cells, indices_hd_cells, indices_conjunctive_cells):
        rm_plotter = RateMapsPlotter(self.exp_dir, wandb_log=self.args.wandb)
        rm_plotter.average_rate_map(rate_maps)
        rm_plotter.metric_histogram(si_r, "Spatial Information (SIr)")
        rm_plotter.rate_maps(
            10, rate_maps, si_r, indices_place_cells, indices_hd_cells, indices_conjunctive_cells
        )


    def rnn_hd_activity(
        self,
        latent_activity, lact_half1, lact_half2,
        thetas, thet_half1, thet_half2,
        positions
    ):
        exp_dir_hd = os.path.join(self.exp_dir, 'hd')
        os.makedirs(exp_dir_hd, exist_ok=True)

        pm_helper = PolarMaps(thetas)
        polar_maps, occ = pm_helper.calculate_polar_maps(latent_activity)
        np.save(os.path.join(exp_dir_hd, 'polar_maps.npy'), polar_maps)

        print("\tCalculating metrics SId and RVL")
        si_d, rvl, rvangle = pm_helper.calculate_metrics(polar_maps.copy(), occ)
        np.save(os.path.join(exp_dir_hd, 'si.npy'), si_d)
        np.save(os.path.join(exp_dir_hd, 'rvl.npy'), rvl)

        pm_stability = None
        try:
            print("\tPolar maps stability")
            hu_half1 = PolarMaps(thet_half1)
            pm_half1, _ = hu_half1.calculate_polar_maps(lact_half1)

            hu_half2 = PolarMaps(thet_half2)
            pm_half2, _ = hu_half2.calculate_polar_maps(lact_half2)

            pm_stability = pm_helper.polar_maps_stability(pm_half1, pm_half2)
            np.save(os.path.join(exp_dir_hd, 'pm_half1.npy'), pm_half1)
            np.save(os.path.join(exp_dir_hd, 'pm_half2.npy'), pm_half2)
            np.save(os.path.join(exp_dir_hd, 'pm_stability.npy'), pm_stability)
        except Exception as e:
            print("\t[-] Polar maps calculation for half of the trajectory failed because\n")
            print(f"\t{e}\n")
            pass

        print("\tCalculating polar maps vs place")
        pm_vs_place = pm_helper.calculate_pm_vs_place(latent_activity, positions, self.args.env_dim)
        pm_vs_place_stability = pm_helper.pm_vs_place_stability(polar_maps, pm_vs_place)
        np.save(os.path.join(exp_dir_hd, 'pm_vs_place.npy'), pm_vs_place)
        np.save(os.path.join(exp_dir_hd, 'pm_vs_place_stability.npy'), pm_vs_place_stability)

        indices_hd_cells = pm_helper.get_hd_cells_indices(polar_maps, si_d, rvl)

        if len(indices_hd_cells) > 0 and self.args.wandb:
            wandb.log({
                WANDB_METRICS_PREFIX+'HD cells stability': float(np.nanmean(pm_stability[indices_hd_cells])),
            })

        return polar_maps, si_d, rvl, rvangle, indices_hd_cells, pm_stability, pm_vs_place, pm_vs_place_stability
    
    def save_hd_plots(self, polar_maps, si_d, rvl, rvangle, indices_place_cells, indices_hd_cells, indices_conjunctive_cells):
        pm_plotter = PolarMapsPlotter(self.exp_dir, PolarMaps.get_thetas_ticks(), wandb_log=self.args.wandb)
        pm_plotter.average_polar_map(polar_maps)
        pm_plotter.metric_histogram(si_d, "Spatial Information (SId)")
        pm_plotter.metric_histogram(rvl, "Rayleigh Vector Length (RVL)")
        pm_plotter.polar_maps(
            10, polar_maps, si_d, rvl, rvangle, indices_place_cells, indices_hd_cells, indices_conjunctive_cells
        )


    def selected_units_analysis(
        self,
        indices_pc, indices_hdc,
        rate_maps, rm_vs_hd, rm_vs_hd_stability,
        polar_maps, pm_vs_place, pm_vs_place_stability
    ):
        indices_conjunctive_cells = np.intersect1d(indices_pc, indices_hdc)
        
        if self.args.wandb:
            wandb.log({
                WANDB_METRICS_PREFIX+'Perc place cells':\
                    float(len(indices_pc)/self.args.latent_dim*100),
                WANDB_METRICS_PREFIX+'Perc HD cells':\
                    float(len(indices_hdc)/self.args.latent_dim*100),
                WANDB_METRICS_PREFIX+'Perc conjunctive cells':\
                    float(len(indices_conjunctive_cells)/self.args.latent_dim*100),
            })

        indices_place_cells = np.setdiff1d(indices_pc, indices_hdc)
        indices_hd_cells = np.setdiff1d(indices_hdc, indices_pc)

        np.save(os.path.join(self.exp_dir, 'indices_place_cells.npy'), indices_place_cells)
        np.save(os.path.join(self.exp_dir, 'indices_hd_cells.npy'), indices_hd_cells)
        np.save(os.path.join(self.exp_dir, 'indices_conjunctive_cells.npy'), indices_conjunctive_cells)

        print(f"\tPlotting rate maps vs HD")
        rm_plotter = RateMapsPlotter(self.exp_dir, wandb_log=self.args.wandb)
        rm_plotter.rate_maps_vs_hd(
            7, rate_maps, rm_vs_hd, rm_vs_hd_stability, 0.3,
            indices_place_cells, indices_hd_cells,
            indices_conjunctive_cells
        )

        print(f"\tPlotting polar maps vs place")
        pm_plotter = PolarMapsPlotter(self.exp_dir, PolarMaps.get_thetas_ticks(), wandb_log=self.args.wandb)
        pm_plotter.polar_maps_vs_place(
            5, pm_vs_place, pm_vs_place_stability, 0.3,
            indices_place_cells, indices_hd_cells,
            indices_conjunctive_cells
        )

        if len(indices_conjunctive_cells) > 0:
            print(f"\tPlotting polar maps and rate maps of conjunctive cells")
            pm_plotter.conjunctive_rms_pms(
                5, rate_maps, polar_maps, indices_conjunctive_cells
            )

        return indices_place_cells, indices_hd_cells, indices_conjunctive_cells

    def pos_hd_decoding(self, lact_h1, lact_h2, pos_h1, pos_h2, thet_h1, thet_h2):
        from sklearn.linear_model import LinearRegression

        labels_h1 = np.concatenate([pos_h1, thet_h1[..., None]], -1)

        lr = LinearRegression()
        lr.fit(lact_h1, labels_h1)
        labels_pred = lr.predict(lact_h2)
        pos_pred, thet_pred = labels_pred[:, :2], labels_pred[:, 2]
        pos_dec_err = np.linalg.norm((pos_h2*100) - (pos_pred*100), axis=1)
        thet_dec_err = np.abs(thet_h2*180/np.pi - thet_pred*180/np.pi)

        np.save(os.path.join(self.exp_dir, 'pos_dec_err.npy'), pos_dec_err)
        np.save(os.path.join(self.exp_dir, 'thet_dec_err.npy'), thet_dec_err)

        if self.args.wandb:
            pos_dec_err_h = np.histogram(pos_dec_err, bins=100, range=(0, 25))
            thet_dec_err_h = np.histogram(thet_dec_err, bins=100, range=(0, 60))
            wandb_log_hist(pos_dec_err_h, 'Pos decoding')
            wandb_log_hist(thet_dec_err_h, 'Thet decoding')

            wandb.log({
                WANDB_METRICS_PREFIX+'Pos decoding': float(np.nanmean(pos_dec_err)),
                WANDB_METRICS_PREFIX+'Thet decoding': float(np.nanmean(thet_dec_err))
            })

        return pos_dec_err, thet_dec_err

    def calculate_sRSA(self, lact, pos):
        from scipy.spatial.distance import cosine, euclidean
        from scipy.stats import spearmanr

        n_exp = lact.shape[0]
        n_timesteps = lact.shape[1]

        d_c = np.zeros((n_exp, n_timesteps, n_timesteps), dtype=np.float32)
        d_e = np.zeros((n_exp, n_timesteps, n_timesteps), dtype=np.float32)
        for exp in range(n_exp):
            for i in range(n_timesteps):
                for j in range(i, n_timesteps):
                    d_c[exp, i, j] = cosine(lact[exp, i], lact[exp, j])
                    d_e[exp, i, j] = euclidean(pos[exp, i], pos[exp, j])
            d_c[exp] = np.triu(d_c[exp]) + np.triu(d_c[exp], k=1).T
            d_e[exp] = np.triu(d_e[exp]) + np.triu(d_e[exp], k=1).T

        
        sRSA_values = np.zeros((n_exp,), dtype=np.float32)
        for exp in range(n_exp):
            sRSA_values[exp] = spearmanr(
                d_e[exp][np.triu_indices(n_timesteps)],
                d_c[exp][np.triu_indices(n_timesteps)]
            ).statistic
        
        np.save(os.path.join(self.exp_dir, 'sRSA_values.npy'), sRSA_values)
        sRSA = float(sRSA_values.mean())
        if self.args.wandb:
            wandb.log({WANDB_METRICS_PREFIX+'sRSA': sRSA})

        return sRSA

    @staticmethod
    def calculate_isomap(lact, pos, seed=0):
        from sklearn.manifold import Isomap

        np.random.seed(seed)
        im = Isomap(
            n_components=2,
            n_neighbors=100,
            metric='cosine',
            n_jobs=-1
        )

        idx_subs = np.random.choice(np.arange(lact.shape[0]), size=7_500, replace=False)

        isomap_emb = im.fit_transform(lact[idx_subs])
        isomap_emb = isomap_emb[:, :3]

        isomap_pos = pos[idx_subs]

        return isomap_emb, isomap_pos

    def save_summary(
        self, vloss_dict, pos_dec_err, thet_dec_err, sRSA,
        indices_pc, n_fields, rm_stability, rm_single_field_dim, rm_vs_hd_stability,
        indices_hdc, pm_stability, pm_vs_place_stability,
        indices_cc,
    ):
        with open(os.path.join(self.exp_dir, f'summary.txt'), 'w') as f:
            s = f'Latent space dimension is {self.args.latent_dim} neurons\n\n'
            f.write(s)
            print(s, end='')

            for k, v in vloss_dict.items():
                s = f'{k}: {v:.5f}\n'
                f.write(s)
                print(s, end='')

            s = f'\nNumber of place cells: {len(indices_pc)}\n'
            f.write(s)
            print(s, end='')
            s = f'Number of HD cells: {len(indices_hdc)}\n'
            f.write(s)
            print(s, end='')
            s = f'Number of conjunctive cells: {len(indices_cc)}\n\n'
            f.write(s)
            print(s, end='')

            s = f'Position decoding error: {np.nanmean(pos_dec_err):.1f}'+\
                f' (SEM: {np.nanstd(pos_dec_err)/np.sqrt(len(pos_dec_err)):.1f})\n'
            f.write(s)
            print(s, end='')
            s = f'HD decoding error: {np.nanmean(thet_dec_err):.0f}'+\
                f' (SEM: {np.nanstd(thet_dec_err)/np.sqrt(len(thet_dec_err)):.0f})\n\n'
            f.write(s)
            print(s, end='')

            s = f'sRSA (Spearman corr): {sRSA:.2f}\n\n'
            f.write(s)
            print(s, end='')

            s = f'Rate maps number of fields: {np.nanmean(n_fields):.2f} '
            f.write(s)
            print(s, end='')
            s = f'({np.nanmean(n_fields[indices_pc]):.2f} in place cells, ' if len(indices_pc) > 0 else '(NA, '
            f.write(s)
            print(s, end='')
            s = f'{np.nanmean(n_fields[indices_cc]):.2f} in conjunctive cells)\n' if len(indices_cc) > 0 else ' NA)\n'
            f.write(s)
            print(s, end='')

            s = f'Rate maps field dimension: {np.nanmean(rm_single_field_dim):.2f} '
            f.write(s)
            print(s, end='')
            s = f'({np.nanmean(rm_single_field_dim[indices_pc]):.2f} in place cells, ' if len(indices_pc) > 0 else '(NA, '
            f.write(s)
            print(s, end='')
            s = f'{np.nanmean(rm_single_field_dim[indices_cc]):.2f} in conjunctive cells)\n\n' if len(indices_cc) > 0 else ' NA)\n\n'
            f.write(s)
            print(s, end='')

            s = f'Rate maps stability 1st vs 2nd half: {np.nanmean(rm_stability):.2f} '
            f.write(s)
            print(s, end='')
            s = f'({np.nanmean(rm_stability[indices_pc]):.2f} in place cells, ' if len(indices_pc) > 0 else ' (NA, '
            f.write(s)
            print(s, end='')
            s = f'{np.nanmean(rm_stability[indices_cc]):.2f} in conjunctive cells)\n' if len(indices_cc) > 0 else ' NA)\n'
            f.write(s)
            print(s, end='')

            s = f'Rate maps stability vs HD: {np.nanmean(rm_vs_hd_stability):.2f} '
            f.write(s)
            print(s, end='')
            s = f'({np.nanmean(rm_vs_hd_stability[indices_pc]):.2f} in place cells, ' if len(indices_pc) > 0 else ' (NA, '
            f.write(s)
            print(s, end='')
            s = f'{np.nanmean(rm_vs_hd_stability[indices_cc]):.2f} in conjunctive cells)\n' if len(indices_cc) > 0 else 'NA)\n\n'
            f.write(s)
            print(s, end='')

            s = f'\nPolar maps stability 1st vs 2nd half: {np.nanmean(pm_stability):.2f}'
            f.write(s)
            print(s, end='')
            s = f' ({np.nanmean(pm_stability[indices_hdc]):.2f} in HD units, ' if len(indices_hdc) > 0 else ' (NA, '
            f.write(s)
            print(s, end='')
            s = f'{np.nanmean(pm_stability[indices_cc]):.2f} in conjunctive cells)\n' if len(indices_cc) > 0 else 'NA)\n'
            f.write(s)
            print(s, end='')

            s = f'Polar maps stability vs place: {np.nanmean(pm_vs_place_stability):.2f}'
            f.write(s)
            print(s, end='')
            s = f' ({np.nanmean(pm_vs_place_stability[indices_hdc]):.2f} in HD units, ' if len(indices_hdc) > 0 else ' (NA, '
            f.write(s)
            print(s, end='')
            s = f'{np.nanmean(pm_vs_place_stability[indices_cc]):.2f} in conjunctive cells)\n' if len(indices_cc) > 0 else 'NA)\n'
            f.write(s)
            print(s, end='')

