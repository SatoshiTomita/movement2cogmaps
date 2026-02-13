from copy import deepcopy
import numpy as np
from scipy import ndimage

from utils.metrics import get_spatial_correlation

SMOOTH_SIGMA = .75


########################
# FILTERS ##############
########################

def primary_filter(field_ratemap, min_bins=0, min_peak_value=0):
    """
    Returns True if field ratemap passes filter criteria, False otherwise

    Args:
    ratemap (np.array): shape (n_samples_pos, n_samples_pos) array with zeros outside field
    min_bins (int): number of non-zero bins in ratemap required to pass
    min_peak_value (float): minimum required maximum value in ratemap to pass
    """
    if np.count_nonzero(field_ratemap) < min_bins:
        return False
    if np.nanmax(field_ratemap) < min_peak_value:
        return False

    return True

def compute_field_stability(
    field_ratemap, rm1, rm2, min_included_value, min_bins
):
    """
    Returns the pearson correlation of two ratemaps at location of field ratemap.
    Excludes all bins that are numpy.nan or < 0 in field ratemap.

    Args:
    ratemap (np.array): shape (n_samples_pos, n_samples_pos) array with np.nan outside field
    rm1 (np.array): (n_samples_pos, n_samples_pos) The first rate map.
    rm2 (np.array): (n_samples_pos, n_samples_pos) The second rate map.
    min_included_value (float): minimum value in ratemap_1 and ratemap_2 for bin to be included
    min_bins (int): minimum number of bins that must remain to compute correlation, else returns numpy.nan

    Returns:
    float: The pearson correlation of the two ratemaps.
    """
    if rm1 is None or rm2 is None:
        return np.nan
    
    field_ratemap = field_ratemap.copy()
    field_ratemap[np.isnan(field_ratemap)] = 0
    return get_spatial_correlation(
        rm1, rm2,
        min_included_value=min_included_value, min_bins=min_bins, mask=(field_ratemap > 0),
    )[0]

def secondary_filter(
    ratemap, max_area_bins, stability_ratemaps,
    min_stability, stability_kwargs
):
    """
    Returns True if ratemap passes filter criteria, False otherwise

    Args:
    ratemap (np.array): shape (n_ybins, n_xbins) array with np.nan outside field
    max_area_bins (int): maximum number of bins greater than 0 in ratemap allowed to pass
    stability_ratemaps (tuple): tuple with two ratemaps that are used for computing the stability
        of a field_ratemap after masking them with the non-numpy.nan elements in field_ratemap
    min_stability (float): minimum required stability in stability_ratemaps to pass
    stability_kwargs (dict): passed on to compute_field_stability

    Returns:
    bool: True if ratemap passes filter criteria, False otherwise
    """
    if np.count_nonzero(ratemap) > max_area_bins:
        return False

    if 'min_included_value' in stability_kwargs and stability_kwargs['min_included_value'] <= 0:
        raise Exception('This module uses 0 values as indication of outside of bin areas.\n'
                        + 'Therefore, min_included_value must be above 0 for stability computation,\n'
                        + 'but is currently {}'.format(stability_kwargs['min_included_value']))
    stability = compute_field_stability(
        ratemap, stability_ratemaps[0], stability_ratemaps[1], **stability_kwargs
    )
    if np.isnan(stability) or (stability < min_stability):
        return False

    return True


########################
# AUXILIARIES ##########
########################

def get_filtered_subfield_ratemaps(ratemap, threshold, primary_filter_kwargs):
    """
    Returns a list containing a copy of ratemap for each field that passes the primary filter
    :py:func:`spatial.fields.primary_filter` where values outside the field are set `0`.

    Args:
    ratemap (np.array): shape (n_ybins, n_xbins)
    primary_filter_kwargs (dict): see :py:func:`spatial.field.primary_filter` keyword arguments.
        filter_kwargs is passed on to that function as `**filter_kwargs` after `ratemap` argument.

    Returns:
    list: field_ratemaps
    """
    # Extract fields with ndimage.label
    field_map = ndimage.label(ratemap > threshold)[0]

    field_nrs = np.unique(field_map)[1:]  # ignores the 0 background field

    # If no fields were detected, return no fields
    if len(field_nrs) == 0:
        return []

    field_ratemaps = []
    for field_nr in field_nrs:
        # create field ratemap
        field_ratemap = ratemap.copy()
        field_ratemap[field_map != field_nr] = 0
        if primary_filter(field_ratemap, **primary_filter_kwargs):
            field_ratemaps.append(field_ratemap)

    return field_ratemaps


def detect_field_candidates(ratemap, base_threshold, threshold_step, primary_filter_kwargs):
    """
    Returns a list of field_candidates that passed primary_filter.

    The returned field_candidates list is a nested list with the following structure:
        - Each element in list contains two elements.
        - The first element is the origin index within field_candidates. This is None for first subfields,
          but int after.
        - The second element is a list of ratemaps with increasingly higher threshold, that all
          have only a single continugous region. The final element in the ratemap list is None,
          if no further subfields were found with the next threshold (none passed the primary_filter).
        - If more than one subfield is found, these are separately appended to field_candidates list following
          the same structure and using the origin index of the previous level where they were detected.

    Args:
    ratemap (np.array): shape (n_ybins, n_xbins) ratemap. Any numpy.nan elements should
        be replaced with zeros before passing ratemap to this function
    base_threshold (float): baseline threshold level from which to start detecting fields
    threshold_step (float): threshold shift at each iteration
    primary_filter_kwargs (dict): see :py:func:`spatial.field.primary_filter` keyword arguments.
        filter_kwargs is passed on to that function as `**filter_kwargs` after `ratemap` argument.

    Returns:
    list: field_candidates
    """

    # Find all subfields in the ratemap at baseline threshold
    subfields = get_filtered_subfield_ratemaps(ratemap, base_threshold, primary_filter_kwargs)

    # If no subfields were found, return an empty list
    if len(subfields) == 0:
        return []

    # Create fields list with the initial subfields
    field_candidates = []
    thresholds = []
    parents = []
    for subfield in subfields:
        field_candidates.append((-1, [subfield]))
        parents.append(-1)
        thresholds.append(base_threshold)

    idx_field = 0
    threshold_curr = thresholds[0]

    # Continuoue the loop until the last idx_field ratemap list ends with None
    while not (field_candidates[-1][1][-1] is None):

        # If current subfield ratemap list ends with None, move to next subfield
        if field_candidates[idx_field][1][-1] is None:
            idx_field += 1
            threshold_curr = thresholds[idx_field]

        # Increase threshold and create a copy of ratemap_curr with the threshold
        threshold_curr += threshold_step
        ratemap_curr = field_candidates[idx_field][1][-1]

        # Find subfields for current field ratemap
        subfields = get_filtered_subfield_ratemaps(ratemap_curr, threshold_curr, primary_filter_kwargs)

        if len(subfields) == 0:
            # If no subfields were found, end the current ratemap list with None
            field_candidates[idx_field][1].append(None)
        elif len(subfields) == 1:
            # If a single field was found, append to current ratemap list
            field_candidates[idx_field][1].append(subfields[0])
        else:
            # If more than one field was found, append these to field_candidates list
            for subfield in subfields:
                field_candidates.append((idx_field, [subfield]))
                parents.append(idx_field)
                thresholds.append(threshold_curr)
            # If more than one field was found, move to next field
            idx_field += 1
            threshold_curr = thresholds[idx_field]

    return field_candidates


def extract_fields_from_field_candidates(field_candidates, secondary_filter_kwargs):
    """
    Returns the field_ratemap with lowest threshold of each field_candidate that
    passes the :py:func:`secondary_filter`.

    Iterates through field_candidates in reverse order of detection in :py:func:`detect_field_candidates`.
    Ignores any field_candiate elements that were the origin of sub-fields that pass secondary_filter.

    Args:
    field_candidates (list): output from :py:func:`detect_field_candidates`
    secondary_filter_kwargs (dict): see :py:func:`spatial.field.secondary_filter` keyword arguments.
        filter_kwargs is passed on to that function as `**filter_kwargs` after `ratemap` argument.

    Returns:
    list: field_ratemaps
    """
    field_ratemap_dicts = []

    # Loop through levels of the field_candidates list starting from the last
    for idx_field in range(len(field_candidates))[::-1]:

        # Get the field_candidate element for the idx_field
        field_candidate = field_candidates[idx_field]

        # Find out if current level field_candidate has any sub-fields already passed the secondary_filter
        n_subfields = 0
        for field_ratemap_dict in field_ratemap_dicts:
            if field_ratemap_dict['level'] == idx_field:
                n_subfields += 1

        # If more than one subfield has been identified for this field_candidate,
        # pass level of this field_candidate to detected subfields and skip processing this field_candidate.
        if n_subfields > 1:
            for field_ratemap_dict in field_ratemap_dicts:
                if field_ratemap_dict['level'] == idx_field:
                    field_ratemap_dict['level'] = field_candidate[0]
            continue

        field_ratemap = None
        # Loop through the ratemaps of this field_candiate
        for field_candidate_ratemap in field_candidate[1][::-1]:

            # field_candidate_ratemap lists can end in None. Ignore these elements.
            if field_candidate_ratemap is None:
                continue

            if secondary_filter(field_candidate_ratemap, **secondary_filter_kwargs):
                # If a ratemap passes the secondary_filter, overwrite the field_ratemap
                # This way final field_ratemap is the one detected with lowest threshold
                # but still passes the secondary_filter.
                field_ratemap = field_candidate_ratemap

        if field_ratemap is None:
            if n_subfields == 1:
                # If no field_ratemap passed the secondary_filter fo this field_candidate,
                # but this field_candidate had one subfield passing through the filter earlier
                # assign the level of that subfield to be the level of current field_candidate.
                subfield_index = [
                    field_ratemap_dict['level']
                    for field_ratemap_dict in field_ratemap_dicts
                ].index(idx_field)
                field_ratemap_dicts[subfield_index]['level'] = field_candidate[0]
        else:
            # If a field_ratemap did pass the secondary_filter, append it to field_ratemap_dicts
            field_ratemap_dicts.append(
                {'ratemap': field_ratemap, 'level': field_candidate[0]}
            )
            if n_subfields == 1:
                # Remove the single subfield of the current field_candidate from field_ratemap_dicts list
                subfield_index = [
                    field_ratemap_dict['level']
                    for field_ratemap_dict in field_ratemap_dicts
                ].index(idx_field)
                del field_ratemap_dicts[subfield_index]

    field_ratemaps = [
        field_ratemap_dict['ratemap'] for field_ratemap_dict in field_ratemap_dicts
        if 'ratemap' in field_ratemap_dict
    ]

    return field_ratemaps


########################
# MAIN FUNCTION ########
########################

def detect_fields(
    ratemap, stability_ratemaps, base_threshold, threshold_step,
    primary_filter_kwargs, secondary_filter_kwargs
):
    """
    Returns a list of copies of input `ratemap` with every value except
    those in field replaced by `numpy.nan`.

    Args:
    ratemap (np.ndarray): shape (n_ybins, n_xbins) ratemap for fields to be detected.
        Values to be ignored should be set to numpy.nan.
    stability_ratemaps (tuple): tuple with two ratemaps that are used for computing the stability
        of a field_ratemap after masking them with the non-numpy.nan elements in field_ratemap
    base_threshold (float): baseline threshold level from which to start detecting fields.
    threshold_step (float): the step in ratemap values for iterative detection of fields.
    primary_filter_kwargs (dict): see :py:func:`spatial.field.primary_filter` keyword arguments.
        filter_kwargs is passed on to that function as `**filter_kwargs` after `ratemap` argument.
    secondary_filter_kwargs (dict): see :py:func:`spatial.field.secondary_filter` keyword arguments.
        `'max_relative_bins'` element is replaced with `'max_area_bins'` computed based on the `ratemap`.
        secondary_filter_kwargs is then passed on to that function as `**filter_kwargs` after `ratemap` argument.

    Returns:
    list: field_ratemaps list of ratemaps where values outside field are numpy.nan
    """
    # Add stability_ratemaps and max_area_bins to a copy of secondary_filter_kwargs
    secondary_filter_kwargs = deepcopy(secondary_filter_kwargs)
    secondary_filter_kwargs['stability_ratemaps'] = stability_ratemaps
    secondary_filter_kwargs['max_area_bins'] = \
        np.sum(~np.isnan(ratemap)) * secondary_filter_kwargs.pop('max_relative_bins')

    # Detect field candidates and extract those that pass all filters
    field_candidates = detect_field_candidates(ratemap, base_threshold, threshold_step, primary_filter_kwargs)
    field_ratemaps = extract_fields_from_field_candidates(field_candidates, secondary_filter_kwargs)

    # Set field_ratemap values of 0 to numpy.nan to indicate outside of field areas
    for i, field_ratemap in enumerate(field_ratemaps):
        field_ratemaps[i][field_ratemap == 0] = np.nan

    return field_ratemaps
