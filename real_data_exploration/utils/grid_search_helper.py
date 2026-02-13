import numpy as np

from real_data_exploration.utils.process_data import (
    calculate_hist_occ, calculate_hist_2d_occ
)
from utils.metrics import get_jensen_shannon

def parametrize_riab_simulations(
    x_all_riab, y_all_riab,
    speed_all_riab,
    rs_all_riab,
    config,
    box_length,
):
    
    occ_pos_riab = calculate_hist_2d_occ(
        x_all_riab, y_all_riab,
        bin_size=box_length/config['POS_BINS'],
        limits=[(0,box_length), (0,box_length)],
        sigma=config['POS_SIGMA_SMOOTHING']
    )

    occ_s_riab = calculate_hist_occ(
        speed_all_riab, config['SPEED_BINS'], config['SPEED_RANGE']
    )
    occ_rs_riab = calculate_hist_occ(
        rs_all_riab, config['RS_BINS'], [np.pi*x for x in config['RS_RANGE']]
    )

    return occ_pos_riab, occ_s_riab, occ_rs_riab


def compare_parameters(
    occ_s_dict, occ_s,
    occ_rs_dict, occ_rs,
    occ_pos_dict, occ_pos,
    tm_fold_dict, tm_fold,
):
    js_s_dict, js_rs_dict = {}, {}
    js_pos_dict, js_tm_dict = {}, {}

    for age in sorted(occ_s_dict.keys()):
        js_s_dict[age] = get_jensen_shannon(occ_s, occ_s_dict[age])
        js_rs_dict[age] = get_jensen_shannon(occ_rs, occ_rs_dict[age])

        js_pos_dict[age] = get_jensen_shannon(occ_pos, occ_pos_dict[age], axis=(0,1))
        js_tm_dict[age] = get_jensen_shannon(tm_fold, tm_fold_dict[age], axis=(-1,-2)).mean()

    return (
        js_s_dict, js_rs_dict,
        js_pos_dict, js_tm_dict,
    )
