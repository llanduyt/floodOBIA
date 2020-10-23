#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Lisa Landuyt
@purpose: Main function for OBIAflood, an unsupervised object-based flood mapping approach
@input: specified in .csv (called as second argument)
@output: .tif and .shp storing classification output are created in directory_output
"""

import sys, os
import pandas as pd
import geopandas as gpd
import numpy as np
import datetime
import rasterio
import rasterstats as rs
import multiprocessing
import pickle

settings_filename = sys.argv[1]
settings = pd.read_csv(os.path.realpath(settings_filename))
settings = settings.replace({np.nan: None})
settings = dict(zip(settings["variable"], settings["value"]))

sys.path.append(os.path.realpath(settings["directory_modules"]))
import ImageFunctions as ipf
import QuickshiftSegmentationFunction as qsf
import ObjectRefinementFunctions as orf
import TiledThresholdingFunctions as ttf
import ClusteringFunctions as cf
import PostProcessingFunctions as ppf
import AccuracyFunctions as af

#%% Settings from CSV
bandnames_mean = ["rVH_mean", "rVV_mean", "fVH_mean", "fVV_mean"]
bandnames_stdev = ["rVV_std", "rVH_std", "fVV_std", "fVH_std"]
# Case specific settings
directory_input = os.path.realpath(settings["directory_input"])
directory_output = os.path.realpath(settings["directory_output"])
sar_filename = os.path.join(directory_input, settings["sar_filename"])
optical_filename = os.path.join(directory_input, settings["optical_filename"])
dem_filename = os.path.join(directory_input, settings["dem_filename"])
lc_filename = os.path.join(directory_input, settings["lc_filename"])
if settings["truth_filename"]:
    truth_filename = os.path.join(directory_input, settings["truth_filename"])
else:
    truth_filename = None
# Processing settings
parallel_processing = bool(settings["parallel_processing"]) # whether to activate parallel processing
num_processes = int(settings["num_processes"]) # number of processes to open in parallel
save_intermediate = bool(settings["save_intermediate"]) # whether to save intermediate results
# Image segmentation settings
ratio = float(settings["ratio"]) # ratio between color- and image-space proximity; higher gives more weight to color-space
maxdist = float(settings["maxdist"]) # max. distance for merging of pixel and nearest neighbor with higher intensity
kernel_window_size = int(settings["kernel_window_size"]) # kernel size for density estimation
# Object refinement settings
t_stdev = float(settings["t_stdev"]) # fraction of st.dev. between pixel and neigbours, used to select similar neighbours
t_conv = int(settings["t_conv"]) # max. number of mergers allowed for convergence
t_shape = float(settings["t_shape"]) # shape threshold (eq. 3) 
nodata_value = float(settings["nodata_value"]) # no data value for feature extraction
tile_size = settings["tile_size"] # size of tiles (px) for parallel object refinement
# Clustering settings
feature_space = settings["feature_space"] # a list of band names or a string referring to a combination of subspaces (see ClusteringFunctions)
num_clusters = int(settings["num_clusters"])
# Cluster classification settings
t_incvv = float(settings["t_incvv"]) # threshold for inc_VV (FV classification)
t_incr = float(settings["t_incr"]) # threshold for inc_R (FV classification)
# Post-processing settings
t_incvv_rg = float(settings["t_incvv_rg"]) # threshold for inc_VV (FV region growing)
t_incr_rg = float(settings["t_incr_rg"]) # threshold for inc_R (FV region growing)
include_singlepol = bool(settings["include_singlepol"]) # whether to include objects satisfying VV or VH condition (OF region growing)
frac_neighbours = float(settings["frac_neighbours"]) # minimal fraction of flooded neighbours (DEM region growing)
t_dem = settings["t_dem"] # statistic of DEM flooded neighbours (DEM region growing)
mmu = int(settings["mmu"]) # minimal mapping unit (px)
lc_types = settings["lc_types"] # land cover types to flag as forest ("all", "open", "closed" or list of CGLS land cover types)
if lc_types[0].isnumeric():
    lc_types = settings["lc_types"].split()
    lc_types = [int(el) for el in lc_types]
del settings

#%% Ancillary functions
def extract_features(image_filename, image_label, stats=["mean"], band_num=1, image_transform=None, nodata=-9999):
    """Run zonal statistics using global var segments"""
    t_start = datetime.datetime.now()
    print("{} - {} - Extracting features for {} based on {}".format(datetime.datetime.now(), os.getpid(), image_label, stats))
    image, image_meta = ipf.tif2ar(image_filename, band=band_num)
    image_transform = image_meta["transform"]
    band_num = 1
    statistics = rs.zonal_stats(segments, image, stats=stats, affine=image_transform, band_num=band_num, nodata=nodata)
    t_end = datetime.datetime.now()
    comp_time = (t_end - t_start).total_seconds()
    print("{} - {} - Feature extraction for {} done after {} sec.!".format(datetime.datetime.now(), os.getpid(), image_label, comp_time))
    return image_label, statistics

def refine_segments_tiled(tile_id, segments):
    """Run tiled object refinement"""
    image_sar, image_sar_meta = ipf.tif2ar(sar_filename)
    return orf.apply_tiled_refiment(tile_id, segments.copy(), image_sar, image_sar_meta["transform"], t_stdev=t_stdev, t_conv=t_conv, t_shape=t_shape, bandnames_mean=bandnames_mean, bandnames_stdev=bandnames_stdev)

def save_topickle(segments, out_filename):
    """Save intermediate result to pickle file"""
    with open(out_filename, "wb") as handle:
        pickle.dump(segments, handle)
        
#%% Import input data
print("{} - Loading input data...".format(datetime.datetime.now()))
image_sar, image_sar_meta, image_sar_bandnames = ipf.tif2ar(sar_filename, return_bandnames=True)
pixel_resolution_x = image_sar_meta["transform"][0] 
pixel_resolution_y = -image_sar_meta["transform"][4]
image_sar_bandnames = ["rVH", "rVV", "fVH", "fVV"]

#%% Image segmentation
print("{} - Image segmentation...".format(datetime.datetime.now()))
t_start = datetime.datetime.now()
segments, segments_raster = qsf.apply_quickseg(np.transpose(image_sar, (1, 2, 0)).copy(), image_sar_bandnames, image_sar_meta, ratio=ratio, maxdist=maxdist, kernel_window_size=kernel_window_size)
t_end = datetime.datetime.now()
comp_time = (t_end - t_start).total_seconds()
print("{} - Segmentation finished after {} sec!".format(datetime.datetime.now(), comp_time))
if save_intermediate:
    save_topickle(segments, os.path.join(directory_output, "Segments.pkl"))

#%% Object refinement
# Removal of NaN segments
print("{} - Removal of NaN segments...".format(datetime.datetime.now()))
t_start = datetime.datetime.now()
nan_DNs = list(np.unique(segments_raster[image_sar[0] == nodata_value]))
nan_segments = segments[segments["DN"].isin(nan_DNs)]
segments.drop(nan_segments.index, inplace=True)
t_end = datetime.datetime.now()
comp_time = (t_end - t_start).total_seconds()
del segments_raster
print("{} - Removal of {} NaN segments finished after {} sec!".format(datetime.datetime.now(), len(nan_segments), comp_time))

# Feature extraction SAR bands 
print("{} - Feature extraction for SAR bands...".format(datetime.datetime.now()))
t_start = datetime.datetime.now()
stats = ["mean", "std"]
if __name__ == "__main__":
    if parallel_processing:
        with multiprocessing.Pool(processes=num_processes, initializer=None) as p:
            result = p.starmap(extract_features, zip([sar_filename]*len(image_sar_bandnames), image_sar_bandnames, \
                         [stats]*len(image_sar_bandnames), [e+1 for e in range(len(image_sar_bandnames))]))
            p.close()
            p.join()
            for band_label, statistics in result:
                for s in stats:
                    colname = "{}_{}".format(band_label, s)
                    segments[colname] = [el[s] for el in statistics]
            del result
    else:
        for band_index, band_label in enumerate(image_sar_bandnames):
            statistics = rs.zonal_stats(segments, image_sar[band_index], stats=stats, affine=image_sar_meta["transform"], nodata=nodata_value)
            for s in stats:
                colname = "{}_{}".format(band_label, s)
                segments[colname] = [el[s] for el in statistics]
            del stats
segments["area"] = segments.area / (pixel_resolution_x * pixel_resolution_y)
t_end = datetime.datetime.now()
comp_time = (t_end - t_start).total_seconds()
print("{} - Feature extraction SAR bands finished after {} sec!".format(datetime.datetime.now(), comp_time))
if save_intermediate:
    save_topickle(segments, os.path.join(directory_output, "Segments_FE.pkl"))
 
# Removal 1-pixel segments
print("{} - Removal of 1-pixel segments...".format(datetime.datetime.now()))
t_start = datetime.datetime.now()
l = len(segments)
segments = orf.remove_singular_segments(segments, image_sar, image_sar_meta["transform"])
t_end = datetime.datetime.now()
comp_time = (t_end - t_start).total_seconds()
print("{} - Removal of {} 1-pixel segments finished after {} sec!".format(datetime.datetime.now(), l-len(segments), comp_time))
if save_intermediate:
    save_topickle(segments, os.path.join(directory_output, "Segments_FE_no1.pkl"))

# Object refinement
print("{} - Object refinement for {} segments...".format(datetime.datetime.now(), len(segments)))
t_start = datetime.datetime.now()
if __name__ == "__main__":
    if parallel_processing:
        # Tile segments    
        tiled_segments = orf.tile_segments(segments)
        i_tiled_segments = np.arange(len(tiled_segments))
        # Refine segments across all available cores        
        with multiprocessing.Pool(processes=num_processes, initializer=None) as p:
            result = p.starmap(refine_segments_tiled, zip(i_tiled_segments, tiled_segments))
            p.close()
            p.join()
        # Merge segments
        segments = gpd.GeoDataFrame(pd.concat([r[0] for r in result], ignore_index=True))
        print("{} segments remaining".format(len(segments)))
        print("{} mergers performed in total".format(np.sum(np.array([r[1] for r in result]))))
        print("{} iterations per tile on average".format(np.mean(np.array([r[2] for r in result]))))
        del result
    else:
        segments = orf.refine_segments(segments, image_sar, image_sar_meta["transform"], t_stdev=t_stdev, t_conv=t_conv, t_shape=t_shape, bandnames_mean=bandnames_mean, bandnames_stdev=bandnames_stdev)
t_end = datetime.datetime.now()
comp_time = (t_end - t_start).total_seconds()
print("{} - Object refinement finished after {} sec!".format(datetime.datetime.now(), comp_time))
if save_intermediate:
    save_topickle(segments, os.path.join(directory_output, "Segments_FE_no1_OR.pkl"))

#%% Feature extraction SAR combinations + optical bands 
print("{} - Feature extraction...".format(datetime.datetime.now()))
t_start = datetime.datetime.now()

# SAR combinations
sar_combinations_filename = os.path.join(directory_output, "SARcombinations.tif")
sar_combinations_names = ["rR", "fR", "incVV", "incVH", "incR"]
if not os.path.exists(sar_combinations_filename):
    print("{} - Calculation of SAR band combinations and saving to file...".format(datetime.datetime.now()))
    ratio_ref = image_sar[image_sar_bandnames.index("rVV")] - image_sar[image_sar_bandnames.index("rVH")]
    ratio_flood = image_sar[image_sar_bandnames.index("fVV")] - image_sar[image_sar_bandnames.index("fVH")]
    increase_vv = image_sar[image_sar_bandnames.index("fVV")] - image_sar[image_sar_bandnames.index("rVV")]
    increase_vh = image_sar[image_sar_bandnames.index("fVH")] - image_sar[image_sar_bandnames.index("rVH")]
    increase_ratio = ratio_flood - ratio_ref
    band_combinations = np.stack([ratio_ref, ratio_flood, increase_vv, increase_vh, increase_ratio])
    del ratio_ref, ratio_flood, increase_vv, increase_vh, increase_ratio
    ipf.ar2tif(band_combinations, sar_combinations_filename, image_sar_meta["crs"], image_sar_meta["transform"], band_index=0, dtype=rasterio.float32, band_names=sar_combinations_names)
    del band_combinations
    
optical_names = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
band_labels = sar_combinations_names + optical_names + ["DEM"] + ["LC"]
band_indices = list(range(1, len(sar_combinations_names)+1)) + list(range(1, len(optical_names)+1)) + [1, 1]
band_filenames = [sar_combinations_filename]*len(sar_combinations_names) \
    + [optical_filename]*len(optical_names) \
    + [dem_filename, lc_filename]
band_stats = [["majority"] if el == "LC" else ["mean"] for el in band_labels]
if __name__ == "__main__":
    if parallel_processing:
        with multiprocessing.Pool(processes=num_processes, initializer=None) as p:
            print("{} processes".format(p._processes))
            result = p.starmap(extract_features, zip(band_filenames, band_labels, band_stats, band_indices))
            p.close()
            p.join()
            for band_label, statistics in result:
                for s in band_stats[band_labels.index(band_label)]:
                    if s == "majority":
                        s_label = "main"
                    else:
                        s_label = s
                    colname = "{}_{}".format(band_label, s_label)
                    segments[colname] = [el[s] for el in statistics]
            del result
    else:
        for bf, bl, bs, bi in zip(band_filenames, band_labels, band_stats, band_indices):
            band_label, statistics = extract_features(bf, bl, stats=bs, band_num=bi)
            for s in bs:
                if s == "majority":
                    s_label = "main"
                else:
                    s_label = s
                colname = "{}_{}".format(band_label, s_label)
                segments[colname] = [el[s] for el in statistics]
t_end = datetime.datetime.now()
comp_time = (t_end - t_start).total_seconds()
print("{} - Feature extraction finished after {} sec!".format(datetime.datetime.now(), comp_time))
if save_intermediate:
    save_topickle(segments, os.path.join(directory_output, "Segments_FE_no1_OR_FE.pkl"))

#%% Threshold calculation
print("{} - Threshold selection...".format(datetime.datetime.now()))
t_start = datetime.datetime.now()
bandnames_threshold = ["fVV", "fVH"]
bandnames_threshold_index = [image_sar_bandnames.index(el) for el in bandnames_threshold]
t_all = ttf.calc_tdict(bandnames_threshold, sar_image=image_sar[bandnames_threshold_index], source=["pixels"], approach=["tiled"], t_method=["KI"], tile_dim=[200, 200], n_final=5, directory_figure=directory_output)
t_vv = t_all["pixels"]["tiled"]["KI"]["fVV"]
t_vh = t_all["pixels"]["tiled"]["KI"]["fVH"]
print("KI threshold for VV and VH: ({}, {})".format(t_vv, t_vh))
del image_sar
t_end = datetime.datetime.now()
comp_time = (t_end - t_start).total_seconds()
print("{} - Threshold selection finished after {} sec!".format(datetime.datetime.now(), comp_time))

#%% K-means clustering
print("{} - Clustering...".format(datetime.datetime.now()))
t_start = datetime.datetime.now()
segments["cluster"] = cf.apply_kmeansclustering(segments, num_clusters, feature_space, random_state=0)
t_end = datetime.datetime.now()
comp_time = (t_end - t_start).total_seconds()
print("{} - Clustering finished after {} sec!".format(datetime.datetime.now(), comp_time))
if save_intermediate:
    save_topickle(segments, os.path.join(directory_output, "Segments_FE_no1_OR_FE_C.pkl"))

#%% Cluster classification
print("{} - Cluster classification...".format(datetime.datetime.now()))
t_start = datetime.datetime.now()
#outfile_base = "Segments_KmClust_{}_{}cl".format(features_set, num_clusters)
#outfile_pickle = os.path.join(directory_output, "{}_classified.pkl".format(outfile_base))
#outfile_fig = os.path.join(directory_output, "KmeansClusterClassification_{}_{}clusters.png".format(features_set, num_clusters))
segments["clusclass"] = cf.classify_clusters(segments, "cluster", t_vv, t_vh, t_incvv=t_incvv, t_incr=t_incr)
t_end = datetime.datetime.now()
comp_time = (t_end - t_start).total_seconds()
print("{} - Cluster classification finished after {} sec!".format(datetime.datetime.now(), comp_time))
if save_intermediate:
    save_topickle(segments, os.path.join(directory_output, "Segments_FE_no1_OR_FE_CC.pkl"))

#%% Post-processing
print("{} - Post-processing...".format(datetime.datetime.now()))
t_start = datetime.datetime.now()
segments["model"] = np.array(segments["clusclass"])
print("{} - RG PW".format(datetime.datetime.now()))
segments = ppf.post_process(segments, "PW", t_vv, t_vh, t_incvv_rg=t_incvv_rg, t_incr_rg=t_incr_rg, output_filename=None)
print("{} - RG OF".format(datetime.datetime.now()))
segments = ppf.post_process(segments, "OF", t_vv, t_vh, t_incvv_rg=t_incvv_rg, t_incr_rg=t_incr_rg, output_filename=None)
print("{} - RG FV".format(datetime.datetime.now()))
segments = ppf.post_process(segments, "FV", t_vv, t_vh, t_incvv_rg=t_incvv_rg, t_incr_rg=t_incr_rg, output_filename=None)
print("{} - RG DEM".format(datetime.datetime.now()))
segments = ppf.post_process(segments, "DEM", t_vv, t_vh, t_incvv_rg=t_incvv_rg, t_incr_rg=t_incr_rg, t_dem=t_dem, frac_neighbours=frac_neighbours, output_filename=None)
print("{} - MMU".format(datetime.datetime.now()))
segments = ppf.apply_mmu(segments, model_field="model", mmu=mmu, output_filename=None)
if save_intermediate:
    save_topickle(segments, os.path.join(directory_output, "Segments_FE_no1_OR_FE_CC_PP.pkl"))

print("{} - FF".format(datetime.datetime.now()))
segments = ppf.flag_forests(segments, lc_types="all", forest_field="forested", output_filename=None)
t_end = datetime.datetime.now()
comp_time = (t_end - t_start).total_seconds()
print("{} - Post-processing finished after {} sec!".format(datetime.datetime.now(), comp_time))
if save_intermediate:
    save_topickle(segments, os.path.join(directory_output, "Segments_FE_no1_OR_FE_CC_PP_FF.pkl"))

#%% Save output
print("{} - Saving output to .shp...".format(datetime.datetime.now()))
segments_outfile = os.path.join(directory_output, "OBIAflood_segments.shp")
segments = segments.set_crs(epsg=image_sar_meta["crs"].to_epsg())
segments.to_file(segments_outfile)

print("{} - Saving output to .tif...".format(datetime.datetime.now()))
raster_outfile = os.path.join(directory_output, "OBIAflood_classification.tif")
classif_meta = image_sar_meta.copy()
classif_meta["dtype"] = rasterio.uint8
classif_meta["count"] = 1
ipf.rasterize_tofile(segments, "model", raster_outfile, classif_meta)

#%% Accuracy
if truth_filename:
    print("{} - Calculating accuracy...".format(datetime.datetime.now()))
    if os.path.exists(truth_filename):
        # Calculate truth column
        band_label = "truth"
        s = "majority"
        image_label, statistics = extract_features(truth_filename, band_label, stats=[s])        
        segments[band_label] = [el[s] for el in statistics]
        # Calculate accuracy
        segments = segments[segments["truth"] != -1] # remove segments for which we do not know truth
        segments.loc[segments["model"] == 4, "model"] = 2 # low lying considered as wet
        segments.loc[segments["model"] == 3, "model"] = 2 # FV considered as wet
        segments.loc[segments["truth"] == 3, "truth"] = 2 # FV considered as wet
        segments.loc[segments["truth"] == 4, "truth"] = 0 # forest considered as dry
        acc = af.calculate_metrics(segments["truth"], segments["model"], average="macro")
        acc_nf = af.calculate_metrics(segments.loc[~segments["forested"], "truth"], segments.loc[~segments["forested"], "model"], average="macro")
        print("Accuracy with forest segments: ", acc)    
        print("Accuracy without forest segments: ", acc_nf)
    else: 
        print("Error! Invalid path: {}".format(truth_filename))

print("{} - End of script!".format(datetime.datetime.now()))
