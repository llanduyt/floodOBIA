#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Lisa Landuyt
@purpose: Ancillary functions for post-classification refinement
"""

import numpy as np
import datetime
import pickle
import pygeos


def post_process(segments, target_class, t_vv, t_vh, t_incvv_rg=1, t_incr_rg=1, include_singlepol=False, t_dem="mean", frac_neighbours=0.5, output_filename=None):
    """Apply region growing post-processing on segments for target_class
    
    Inputs:
    segments: geopandas GeoDataFrame
        Segments considered for post-processing.
    target_class: str
        Target class for post-processing, must be one of "PW", "OF", "FV" or "DEM".
    t_vv: float
        VV threshold, used for post-processing PW and OF.
    t_vh: float
        VH threshold, used for post-processing PW and OF.
    t_incvv_rg: float (default=1.0)
        incVV threshold, used for post-procesing FV
    t_incr_rg: float (default=1.0)
        incVH threshold, used for post-procesing FV.
    include_singlepol: bool (default=False)
        Whether to include segments that only satisfy one of VV and VH conditions in PW/OF post-processing.
    t_dem: str (default="mean")
        Statistic to calculate on flooded neighbours for post-processing DEM, must be one of "min", "max" or "mean".
    frac_neighbours: float (default=0.5)
        Minimum fraction of neighbours that should be flooded to be considered for DEM post-processing.
    output_filename: str or None (default=None)
        If not None, output is pickled to specified path.
    Ouputs:
    segments: geopandas GeoDataFrame
        Segments after post-processing.
    """
    if target_class == "PW":
        target_label = 1
        seeds_label = [1]
        source_label = [0, 2, 3]
    elif target_class == "OF":
        target_label = 2
        seeds_label = [1, 2, 3]
        source_label = [0, 3]
    elif target_class == "FV":
        target_label = 3
        seeds_label = [2, 3]
        source_label = [0]
    elif target_class == "DEM":
        target_label = 4
        seeds_label = [1, 2, 3]
        source_label = [0]
    # Reset index and create spatial index
    segments.reset_index(drop=True, inplace=True)
    segments_sindex = pygeos.STRtree(segments.geometry.values.data)
    # Select seeds
    seed_segments = segments.loc[segments["model"].isin(seeds_label)]
    print("{} - {} seeds".format(datetime.datetime.now(), len(seed_segments)))
    # Perform RG
    num_changed = 10
    num_changed_total = 0
    it = 0
    while num_changed > 0:
        it += 1
        print("{} - Iteration {}...".format(datetime.datetime.now(), it))
        # Select neighbours
        _, i_neighbours = segments_sindex.query_bulk(seed_segments.geometry.values.data, predicate="touches")
        i_neighbours = i_neighbours[~np.isin(i_neighbours, seed_segments.index)]
        seed_neighbours = segments.iloc[i_neighbours] # Could still have inaccuracies because based on envelopes
        # From neighbours, select those that are in source_label and fulfill the expresssion
        seg_candidates = seed_neighbours.loc[seed_neighbours["model"].isin(source_label)]
        if target_class == "PW":
            seg_selected = seg_candidates.loc[(seg_candidates["fVV_mean"] < t_vv) & (seg_candidates["fVH_mean"] < t_vh) & (seg_candidates["rVV_mean"] < t_vv) & (seg_candidates["rVH_mean"] < t_vh)]
        elif target_class == "OF":
            if include_singlepol:
                seg_selected = seg_candidates.loc[(seg_candidates["fVV_mean"] < t_vv) | (seg_candidates["fVH_mean"] < t_vh)]
            else:
                seg_selected = seg_candidates.loc[(seg_candidates["fVV_mean"] < t_vv) & (seg_candidates["fVH_mean"] < t_vh)]
        elif target_class == "FV":
            seg_selected = seg_candidates.loc[(seg_candidates["incVV_mean"] >= t_incvv_rg) & (seg_candidates["incR_mean"] >= t_incr_rg)]
        elif target_class == "DEM":
            i_seg, i_seg_neighbours = segments_sindex.query_bulk(seg_candidates.geometry.values.data, predicate="touches")
            # Select based on fraction of neighbouring flooded segments
            candidates_neighbours_indices = np.array([i_seg_neighbours[i_seg == i] for i in np.arange(len(seg_candidates))]) # group neighbours per seg in seg_candidates      
            candidates_frac_floodedneighbours = np.array([np.sum(segments.iloc[i].model > 0) / len(i) for i in candidates_neighbours_indices])
            seg_candidates = seg_candidates[candidates_frac_floodedneighbours >= frac_neighbours]
            candidates_neighbours_indices = candidates_neighbours_indices[candidates_frac_floodedneighbours >= frac_neighbours]
            # Select based on elevation compared to flooded neighbours
            if t_dem == "mean":
                candidates_dem_floodedneighbours = [np.mean(segments.iloc[i]["DEM_mean"]) for i in candidates_neighbours_indices]
            elif t_dem == "min":
                candidates_dem_floodedneighbours = [np.min(segments.iloc[i]["DEM_mean"]) for i in candidates_neighbours_indices]
            elif t_dem == "max":
                candidates_dem_floodedneighbours = [np.max(segments.iloc[i]["DEM_mean"]) for i in candidates_neighbours_indices]
            seg_selected = seg_candidates[seg_candidates["DEM_mean"] < candidates_dem_floodedneighbours]            
        # Update model state of selected neighbours
        segments.loc[seg_selected.index, "model"] = target_label
        num_changed = len(seg_selected)
        num_changed_total += num_changed
        # Update seeds for next iteration
        seed_segments = segments.loc[segments["model"].isin(seeds_label)]
    print("{} - RG for {} done! Changed state of {} segments.".format(datetime.datetime.now(), target_class, num_changed_total))
    # Save updated model output
    if output_filename is not None:
        print("{} - Saving RG output of {} to {}".format(datetime.datetime.now(), target_class, output_filename))
        with open(output_filename, "wb") as handle:
            pickle.dump(segments["model"], handle)
    # Return
    return segments


def apply_mmu(segments, model_field="model", mmu=20, image_pixel_size=100, output_filename=None):
    """Apply a minimal mapping unit
    
    Input:
    segments: geopandas GeoDataFrame
        Segments to consider for MMU.
    model_field: str (default="model")
        Name of column that stores the modeled classification.
    mmu: int (default=20)
        Minimum size of objects to be retained (expressed as number of pixels).
    image_pixel_size: float (default=100)
        Size of image pixels (expressed in m2).
    output_filename: str or None (default=None)
        If not None, output is pickled to specified path.
    Ouputs:
    segments: geopandas GeoDataFrame
        Segments after removal of small objects.
    """
    target_label = 0
    # Calculate unified flood segments
    segments_flood_unified = segments[segments[model_field] > 0].unary_union
    if segments_flood_unified.geom_type == "MultiPolygon":
        segments_flood = list(segments_flood_unified)
    else:
        segments_flood = [segments_flood_unified]
    # Detect small flood segments and find corresponding segment DN's
    segments_flood_small = [seg for seg in segments_flood if seg.area < mmu*image_pixel_size]
    segments_flood_small_dns = [list(segments.loc[segments["geometry"].within(seg), "DN"]) for seg in segments_flood_small] # TO DO: increase speed?
    segments_flood_small_dns = [item for sublist in segments_flood_small_dns for item in sublist]
    # Update model field
    print("Small objects detected: {}".format(len(segments_flood_small_dns)))
    segments.loc[segments["DN"].isin(segments_flood_small_dns), model_field] = target_label
    # Save updated model output
    if output_filename is not None:
        print("{} - Saving output of MMU to {}".format(datetime.datetime.now(), output_filename))
        with open(output_filename, "wb") as handle:
            pickle.dump(segments["model"], handle)
    # Return
    return segments


def flag_forests(segments, lc_types="all", model_field="model", forest_field="forested", output_filename=None):
    """Flag forest segments
    
    Inputs:
    segments: geopandas GeoDataFrame
        Segments considered for forest flagging.
    lc_types: str or list (default="all")
        List of CGLS forest types or one of "all", "closed" or "open", referring to all, only closed and only open forest.
    model_field: str (default="forested")
        Name of column that stores the classification.
    forest_field: str (default="forested")
        Name of column that will store the information on forest cover.
    output_filename: str or None (default=None)
        If not None, output is pickled to specified path.
    Ouputs:
    segments: geopandas GeoDataFrame
        Segments after forest flagging.
    """
    segments[forest_field] = [False] * len(segments)
    # Select LC types
    if isinstance(lc_types, str):
        if lc_types == "all":
            lc_types = list(np.arange(111,117)) + list(np.arange(121,127))
        elif lc_types == "closed":
            lc_types = list(np.arange(111,117))
        elif lc_types == "open":
            lc_types = list(np.arange(121,127))
    elif not isinstance(lc_types, list):
        print('Error! lc_types should be of type list or equal to "all", "open" or closed". Now {}. Switching to default "all".'.format(lc_types))
        lc_types = list(np.arange(111,117)) + list(np.arange(121,127))
    # Select dry segments that have suited LC type
    segments.loc[segments["LC_main"].isin(lc_types), forest_field] = True
    segments_forest_dry = segments.loc[(segments[forest_field]) & (segments[model_field] == 0)]
    # Update model field selected segments
    print("'Dry' forest objects detected: {}, ".format(len(segments_forest_dry)))
    segments.loc[segments_forest_dry.index, model_field] = True
    # Save updated model output
    if output_filename:
        print("{} - Saving output of forest-flagging to {}".format(datetime.datetime.now(), output_filename))
        with open(output_filename, "wb") as handle:
            pickle.dump(segments[[model_field, forest_field]], handle)
    # Return
    return segments
                