# -*- coding: utf-8 -*-
"""
@author: Lisa Landuyt
@purpose: Ancillary functions for object refinement
"""

import os
import numpy as np
import datetime
import pygeos
import shapely
import geopandas as gpd
import pickle
import rasterstats as rs


def update_segment_properties(segment, image_stack, image_transform, bandnames_mean, bandnames_stdev):
    """ Update area and image properties of segment
    
    Inputs:
    segment: pandas Series
        Segment for which to update properties.
    image_stack: nd array
        Image to retriev properties from.
    image_transform: Affine
        Transformation from pixel to geographic coordinates.
    bandnames_mean: list
        Column names to store image band means (same order as bands in image_stack).
    bandnames_stdev: list
        Column names to store image band st. devs. (same order as bands in image_stack).
    Outputs:
    segment: pandas Series
        Segment with updated properties.    
    """
    # update pixel properties of merged segment
    statistics = [rs.zonal_stats(segment.geometry, image_band, affine=image_transform, stats=['mean', 'std'], nodata=-999) for image_band in image_stack]
    segment[bandnames_mean] = [el[0]['mean'] for el in statistics]
    segment[bandnames_stdev] = [el[0]['std'] for el in statistics]
    # update area of merged segment
    segment["area"] = segment.geometry.area / 100
    return segment

def merge_similar_segments(center_segment, similar_segments, image_stack, image_transform, bandnames_mean, bandnames_stdev, t_shape=None):
    """ Merge segments if shape constraint is fulfilled

    Inputs:
    center_segment: pandas Series
        Segment that will be expanded.
    similar_segments: pandas Series or DataFrame
        Segments used to expand center_segment.
    image_stack: nd array
        Image to retriev properties from.
    image_transform: Affine
        Transformation from pixel to geographic coordinates.
    bandnames_mean:
        Column names to store image band means (same order as bands in image_stack).
    bandnames_stdev:
        Column names to store image band st. devs. (same order as bands in image_stack).
    t_shape: int, float or None (default=None)
        If not None, maximum Perimeter/sqrt(Area) to be considered for merger.
    Outputs:
    center_segment: pandas Series
        Expanded segment.
    merged: bool
        True if merger took place.
    """
    geometry_updated = gpd.GeoSeries(similar_segments.append(center_segment).geometry).unary_union
    if not t_shape or (geometry_updated.length/np.sqrt(geometry_updated.area) < t_shape):
        # update geometry of merged segment
        center_segment["geometry"] = geometry_updated
        # update segment properties
        center_segment = update_segment_properties(center_segment, image_stack, image_transform, bandnames_mean, bandnames_stdev)
        merged = True
    else:
        merged = False
    return center_segment, merged

def refine_segments(segments, image_stack, image_transform, t_stdev=1, t_conv=10, t_shape=None, bandnames_mean=["rVH_mean", "rVV_mean", "fVH_mean", "fVV_mean"], bandnames_stdev=["rVV_std", "rVH_std", "fVV_std", "fVH_std"]):
    """ Refine objects by merging similar ones
    
    Inputs:
    segments: geopandas GeoDataFrame
        Segments to consider for refinement.
    image_stack: nd array
        Image to retriev properties from.
    image_transform: Affine
        Transformation from pixel to geographic coordinates.
    t_stdev: int or float (default=1)
        Fraction of min. st. dev. that serves as max for st. dev. difference, in order to allow merger. 
    t_conv: int or float (default=10)
        Max. number of changes allowed to converge.
    t_shape: int, float or None (default=None)
        If not None, maximum Perimeter/sqrt(Area) to be considered for merger.
    bandnames_mean: list (default=["rVH_mean", "rVV_mean", "fVH_mean", "fVV_mean"])
        Column names to store image band means (same order as bands in image_stack).
    bandnames_stdev: list (default=["rVV_std", "rVH_std", "fVV_std", "fVH_std"])
        Column names to store image band st. devs. (same order as bands in image_stack).
    Outputs:
    segments: geopandas GeoDataFrame
        Refined segments.
    num_mergers: int
        Total number of mergers performed.
    its: int
        Number of iterations needed for convergence.
    """
    segments["removed"] = [False] * len(segments)
    segments["DNparent"] = [None] * len(segments)
                
    b = len(bandnames_mean)
    num_mergers = 1e6
    num_mergers_it = []
    num_segments_it = []
    its = 0
    while num_mergers > t_conv:
        segments.reset_index(drop=True, inplace=True)
        segments_sindex = pygeos.STRtree(segments.geometry.values.data)
        left, right = segments_sindex.query_bulk(segments.geometry.values.data, predicate="touches")
        num_mergers = 0
        its += 1
        for seg_index in segments.index:
            seg = segments.loc[seg_index]
            if not seg.removed:
                # Select neighbouring segments
                possible_neighbours = segments.iloc[right[left == seg_index]]
                if len(possible_neighbours) > 0:
                    while np.sum(possible_neighbours["removed"]) > 0:
                        parent_dn = possible_neighbours.loc[possible_neighbours["removed"], "DNparent"]
                        possible_neighbours = possible_neighbours[~possible_neighbours["removed"]]
                        possible_neighbours = possible_neighbours.append(segments[segments["DN"].isin(parent_dn)]) 
                    try:
                        neighbours = possible_neighbours[[type(el) is not shapely.geometry.Point for el in possible_neighbours.geometry.buffer(0).intersection(seg.geometry.buffer(0))]]
                    except:
                        print("Error when filtering neighbours!")
                        with open(os.path.join(os.getcwd(), "Dump_error_objectrefinement.pkl"), "wb") as handle:
                            pickle.dump([seg, possible_neighbours], handle)
                    # Calculate difference between means and min. st. dev. for each pair of seg - neighbour
                    diff = abs(neighbours[bandnames_mean]- seg[bandnames_mean]) # difference of means current segment and neighbours
                    stdev_min = np.minimum(neighbours[bandnames_stdev], seg[bandnames_stdev]) # min st. dev. for each pair of (neighbour, current segment)
                    # Select segments to merge
                    similar_segments = neighbours[np.sum(np.array(diff) < np.array(t_stdev*stdev_min), axis=1) == b] # similar segments are those for which the summed abs. difference < min. st. dev.
                    if len(similar_segments) > 0:
                        # Merge segments (if T_shape allows)
                        seg_updated, merged = merge_similar_segments(seg.copy(), similar_segments, image_stack, image_transform, bandnames_mean, bandnames_stdev, t_shape)
                        if merged:
                            # Update segment in gdf
                            segments.loc[seg_index] = seg_updated
                            # Mark merged segments for removal
                            segments.loc[similar_segments.index, "removed"] = True
                            segments.loc[similar_segments.index, "DNparent"] = seg_updated["DN"]
                            num_mergers += 1
                    del neighbours
                else:
                    print("{} - {} - Isolated segment: {}({})!".format(datetime.datetime.now(), os.getpid(), seg_index, seg["DN"]))
        segments.drop(segments[segments["removed"]].index, axis=0, inplace=True)
        num_mergers_it.append(num_mergers)   
        num_segments_it.append(len(segments))
    num_mergers = np.sum(np.array(num_mergers_it))
    print("{} - {} - Refinement done after {} iterations: {} mergers, {} segments remaining.".format(datetime.datetime.now(), os.getpid(), its, num_mergers, len(segments)))
    return segments.drop(["removed", "DNparent"], axis=1), num_mergers, its

def remove_singular_segments(segments, image_stack, image_transform, t_shape=None, bandnames_mean=["rVH_mean", "rVV_mean", "fVH_mean", "fVV_mean"], bandnames_stdev=["rVV_std", "rVH_std", "fVV_std", "fVH_std"]):
    """Remove 1-pixel segments from GeoDataFrame
    
    Inputs:
    segments: geopandas GeoDataFrame
        GeoDataFrame to remove 1-pixel segments from.
    image_stack: nd array
        Image to retriev properties from.
    image_transform: Affine
        Transformation from pixel to geographic coordinates.
    t_shape: int, float or None (default=None)
        If not None, maximum Perimeter/sqrt(Area) to be considered for merger.
    bandnames_mean: list (default=["rVH_mean", "rVV_mean", "fVH_mean", "fVV_mean"])
        Column names to store image band means (same order as bands in image_stack).
    bandnames_stdev: list (default=["rVV_std", "rVH_std", "fVV_std", "fVH_std"])
        Column names to store image band st. devs. (same order as bands in image_stack).
    Ouputs:
    segments: geopandas GeoDataFrame
        GeoDataFrame without 1-pixel segments.
    """
    segments["removed"] = [False] * len(segments)
    segments.reset_index(drop=True, inplace=True)

    singular_segments = segments[segments["area"] == 1]
    singular_segments_index = list(singular_segments.index)
    print("Found {} 1-pixel segments".format(len(singular_segments)))

    segments_sindex = pygeos.STRtree(segments.geometry.values.data)
    left, right = segments_sindex.query_bulk(singular_segments.geometry.values.data, predicate="touches")

    for seg_index, seg in singular_segments.iterrows():
        possible_neighbours = segments.iloc[right[left == singular_segments_index.index(seg_index)]]
        possible_neighbours = possible_neighbours[~possible_neighbours["removed"]] # remove removed segments from selection
        possible_neighbours = possible_neighbours[possible_neighbours["area"] > 1] # remove 1-pixel segments from selection
        neighbours = possible_neighbours[[type(el) is not shapely.geometry.Point for el in possible_neighbours.geometry.buffer(0).intersection(seg.geometry.buffer(0))]]
        if len(neighbours) == 0: # only look at 2-connectivity if no neighbours in 1-connectivity (should not happen)
            print("No neighbours found for segment {}. Swithching to 2-connectivity".format(seg_index))
            neighbours = possible_neighbours
        diff = abs(neighbours[bandnames_mean]- seg[bandnames_mean]) # difference of means current segment and neighbours
        most_similar_segment = neighbours[np.sum(np.array(diff), axis=1) == np.min(np.sum(np.array(diff), axis=1))].iloc[0]
        seg_updated, merged = merge_similar_segments(most_similar_segment.copy(), seg.copy(), image_stack, image_transform, bandnames_mean, bandnames_stdev, t_shape)
        segments.loc[most_similar_segment.name] = seg_updated
        segments.loc[seg_index, "removed"] = True
    return segments[segments["removed"] == False].drop("removed", axis=1)   

def tile_segments(segments, tile_size=(5000,5000)):
    """Split up segments in subsets
    
    Inputs:
    segments: geopandas GeoDataFrame
        Segments to split up.
    tile_size: tuple
        Size of subset footprints (geographic coordinates).
    Ouputs:
    tiled_segments: list
        List of segments subsets.
    """
    bounds = segments.total_bounds
    nct, nrt = np.ceil((bounds[2:] - bounds[:2]) / tile_size).astype('int')
    # Calculate tile bboxes
    tile_numbers = np.arange(nrt*nct)
    tile_ir = (np.floor(tile_numbers / nct)).astype(int)
    tile_ic = tile_numbers % nct
    xmin = bounds[0] + tile_ic * tile_size[1]
    ymax = bounds[3] - tile_ir * tile_size[0]
    xmax = xmin + tile_size[0]
    ymin = ymax - tile_size[1]
    tile_boxes = pygeos.creation.box(xmin, ymin, xmax, ymax)
    # Spatial query on segments 
    segments_sindex = pygeos.STRtree(segments.geometry.values.data)
    left, right = segments_sindex.query_bulk(tile_boxes, predicate="intersects")
    # Divide segments amongst tiles
    segments["taken"] =  [False] * len(segments)   
    tiled_segments = []
    for tile_i in tile_numbers:
        segments_subset = segments.iloc[right[left == tile_i]].copy()
        segments_subset.drop(segments_subset[segments_subset["taken"]].index, inplace=True)
        segments.loc[segments_subset.index, "taken"] = True
        segments_subset.reset_index(drop=True, inplace=True)
        tiled_segments.append(segments_subset.drop("taken", axis=1))
    # Return
    return tiled_segments

def apply_tiled_refiment(tile_id, segments, image_stack, image_transform, t_stdev=1, t_conv=10, t_shape=None, bandnames_mean=["rVH_mean", "rVV_mean", "fVH_mean", "fVV_mean"], bandnames_stdev=["rVV_std", "rVH_std", "fVV_std", "fVH_std"]):
    """ Apply object refinement using tiled approach for increased speed
    
    Inputs:
    tile_id
    segments: geopandas GeoDataFrame
        GeoDataFrame to remove 1-pixel segments from.
    image_stack: nd array
        Image to retriev properties from.
    image_transform: Affine
        Transformation from pixel to geographic coordinates.
    t_stdev: int or float (default=1)
        Fraction of min. st. dev. that serves as max for st. dev. difference, in order to allow merger. 
    t_conv: int or float (default=10)
        Max. number of changes allowed to converge.
    t_shape: int, float or None (default=None)
        If not None, maximum Perimeter/sqrt(Area) to be considered for merger.
    bandnames_mean: list (default=["rVH_mean", "rVV_mean", "fVH_mean", "fVV_mean"])
        Column names to store image band means (same order as bands in image_stack).
    bandnames_stdev: list (default=["rVV_std", "rVH_std", "fVV_std", "fVH_std"])
        Column names to store image band st. devs. (same order as bands in image_stack).
    Ouputs:
    segments: geopandas GeoDataFrame
        Refined segments.
    num_mergers: int
        Total number of mergers performed.
    its: int
        Number of iterations needed for convergence.
    """
    print("{} - Refinement for tile {} starting in {}".format(datetime.datetime.now(), tile_id, os.getpid()))
    return refine_segments(segments, image_stack, image_transform,t_stdev=t_stdev, t_conv=t_conv, t_shape=t_shape, bandnames_mean=bandnames_mean, bandnames_stdev=bandnames_stdev)

