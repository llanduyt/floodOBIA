#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Lisa Landuyt
@purpose: Ancillary functions for clustering
"""

import os
import datetime
import numpy as np
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
import sklearn.cluster as cl
from sklearn.neighbors import NearestNeighbors
import pickle

import PlottingFunctions as pf

def get_point_density(my_point, X, nbrs):
    """Retrieve density of single point"""
    bandwidth = nbrs.get_params()['radius']
    i_nbrs = nbrs.radius_neighbors([my_point], bandwidth, return_distance=False)[0]
    return len(i_nbrs)

def prepare_data(segments, feature_space):
    """Prepare data for clustering (FS selection + normalization)
    
    Inputs:
    segments: string or geopandas geodataframe
        Path to segments dataframe or segments dataframe.
    feature_space: list or str
        List of band names to include in FS or keyword describing FS. Keyword should be constructed
        as "bands_stats", where bands can be one or a combination of "SARflood", "SAR", "wR", "wC",
        "incF", "wHAND", "o3", "opt", "lc" or "lc_frac" and stats should be "mean" or "all" (with "all"
        referring to mean + std).
    Outputs:
    X: nd array
        Array of FS data, scaled to 0 mean and unit variance.
    labels: list
        List of feature (band) labels.
    """
    if type(segments) == str:
        print("{} - Loading segments from file...".format(datetime.datetime.now()))
        segments = gpd.read_file(segments)
    elif type(segments) != gpd.geodataframe.GeoDataFrame:
        print("Error! Input segments of wrong type ({} vs. gpd.geodataframe.GeoDataFrame)".format(type(segments)))
        return None

    if type(feature_space) is list:
        feature_names = feature_space
    elif "_" not in feature_space:
        print("Error! Features set not recognized: {}".format(feature_space))
        return None, None
    else:
        features = feature_space[:feature_space.index("_")]
        stat = feature_space[feature_space.index("_")+1:]
        feature_names = []
        # SAR bands
        if "SARflood" in features:
            feature_names += ['fVV_mean', 'fVH_mean']
            if "wR" in features:
                feature_names.append('fR_mean')
            if ("wC" in features) or ("incF" in features):
                print("Error! wC and incF options only possible when 2 time steps are considered.")
                return None, None
            if stat == "all":
                feature_names += ['fVV_std', 'fVH_std']
                if "wR" in features:
                    feature_names.append('fR_std')
        elif "SAR" in features:
            feature_names += ['rVV_mean', 'rVH_mean', 'fVV_mean', 'fVH_mean']
            if stat == "all":
                feature_names += ['rVV_std', 'rVH_std', 'fVV_std', 'fVH_std']
            # SAR band combinations
            if "wR" in features:
                feature_names += ['rR_mean', 'fR_mean']
                if stat == "all":
                    feature_names += ['rR_std', 'fR_std']
            if "incF" in features:
                feature_names += ['incVV_mean', 'incVH_mean', 'incR_mean']
            elif "wC" in features:
                feature_names += ['rR_mean', 'fR_mean', 'incVV_mean', 'incVH_mean', 'incR_mean']
        # HAND
        if "wHAND" in features:
            feature_names.append('HAND_mean')
            if stat == "all":
                feature_names.append("HAND_std")
        # optical bands
        if "opt" in features:
            feature_names += ['B2_mean', 'B3_mean', 'B4_mean', 'B5_mean', 'B6_mean', 'B7_mean', 'B8_mean', 'B8A_mean', 'B11_mean', 'B12_mean']
            if stat == "all":
                feature_names += ['B2_std', 'B3_std', 'B4_std', 'B5_std', 'B6_std', 'B7_std', 'B8_std', 'B8A_std', 'B11_std', 'B12_std']
        elif "o3" in features:
            feature_names += ['B4_mean', 'B8_mean', 'B12_mean']
        # LC bands
        if "lcfrac" in features:
            feature_names += ['fBare_mean', 'fGras_mean', 'fCrop_mean', 'fShru_mean', 'fTree_mean', 'fPW_mean', 'fSW_mean', 'fUrba_mean']
        elif "lc" in features:
            feature_names.append("LC_main")

    return sklearn.preprocessing.scale(segments[feature_names]), feature_names


def apply_kmeansclustering(segments, n_clusters, feature_space, random_state=0, init='k-means++', directory_output=None, segments_name="Segments", return_labels=False):
    """
    Apply K-means clustering on segments considering feature_space
    
    Inputs:
    segments: geopandas GeoDataFrame
        Objects to consider for clustering.
    n_clusters: int
        Number of clusters.
    feature_space: list or str
        List of band names to include in FS or keyword describing FS. Keyword should be constructed
        as "bands_stats", where bands can be one or a combination of "SARflood", "SAR", "wR", "wC",
        "incF", "wHAND", "o3", "opt", "lc" or "lc_frac" and stats should be "mean" or "all" (with "all"
        referring to mean + std).
    random_state: int or None (default=0)
        Determines random number generation for centroid initialization. Use an int to make the randomness deterministic.
    init: 'k-means++', 'random' or ndarray (default='k-means++')
        K-means initialization method (see sklearn.cluster.KMeans docs).
    directory_output: str or None (default=None)
        If not None, clustering output (+ comp. time) and scatter plot will be saved to this directory.
    segments_name: str (default='Segments')
        Prefix to use when saving output.
    return_labels: bool (default=False)
        Whether to return the labels of the feature space.
    Outputs:
    segments_cluster: geopandas Series
        Cluster labels of all objects.
    labels (if return_labels==True): list
        List of feature (band) labels.
    """
    X, labels = prepare_data(segments, feature_space)
    if X is None:
        return None

    ts = datetime.datetime.now()
    cluster = cl.KMeans(n_clusters=n_clusters, init=init, random_state=random_state)
    cluster = cluster.fit(X)
    segments["cluster"] = cluster.labels_
    te = datetime.datetime.now()
    
    if directory_output:    
        outfile_pickle = os.path.join(directory_output, "{}_KmClust_{}_{}cl.pkl".format(segments_name, feature_space, n_clusters))
        print("{} - Saving cluster output to {}...".format(datetime.datetime.now(), outfile_pickle))
        with open(outfile_pickle, "wb") as handle:
            pickle.dump([segments["cluster"], (te-ts).total_seconds()], handle)

        outfile_fig = os.path.join(directory_output, "{}_KmClust_{}_{}cl.png".format(segments_name, feature_space, n_clusters))
        print("{} - Plotting clusters and saving to {}...".format(datetime.datetime.now(), outfile_fig))
        colors = ['coral', 'lightblue', 'blue', 'lightgreen', 'green', 'grey', 'purple',
                      'yellow', 'red', 'pink', 'saddlebrown', 'cyan', 'violet', 'olive']
        labels_mean = [el for el in labels if el.endswith("mean") and ("VV" in el or "VH" in el)]
        fig, ax = pf.makeScatterSubplotsObjects(segments, labels_mean, colorlabel="cluster", cmap=mpl.colors.ListedColormap(colors[:n_clusters]))
        if fig:
            fig.suptitle('Object intensity means per cluster', fontsize=16)
            plt.savefig(outfile_fig)
            plt.close(fig)

    if return_labels:
        return segments["cluster"], labels
    return segments["cluster"]

def classify_clusters(segments, cluster_column, t_vv, t_vh, t_incvv=1, t_incr=1, image_mode="dual", output_filename=None, fig_filename=None):
    """ 
    Classify clusters based on their centroids
    
    Inputs:
    segments: geopandas GeoDataFrame
        Objects to consider for clustering.
    cluster_column: str
        Label of column (added to gdf) containing cluster classification.
    t_vv: float
        VV threshold for classification of PW/OF.
    t_vh: float
        VH threshold for classification of PW/OF.
    t_incvv: float (default=1)
        incVV threshold for classification of FV.
    t_incr: float (default=1)
        incR threshold for classification of FV.
    image_mode: 'single', or 'dual' (default='dual')
        Label indicated whether a single or two scenes are considered for classification.
    output_filename: str or None (default=None)
        If not None, path to save cluster classification output (as pickle).
    fig_filename: str or None (default=None)
        If not None, path to save scatter plot colored by cluster class.
    Outputs:
    segments_cluster_class: geopandas Series
        Cluster classification labels of all objects (0=DL, 1=PW, 2=OF, 3=FV).
    """
    # Gather features used for classification
    flood_vh_label = "fVH_mean"
    flood_vv_label = "fVV_mean"
    ref_vh_label = "rVH_mean"
    ref_vv_label = "rVV_mean"
    inc_vv_label = "incVV_mean"
    inc_r_label = "incR_mean"
    columns_mean = [flood_vh_label, flood_vv_label, ref_vh_label, ref_vv_label, inc_vv_label, inc_r_label]
    
    # Calculate cluster centroids
    cluster_labels = np.unique(segments[cluster_column])
    num_clusters = len(np.unique(cluster_labels))
    cluster_centroids = np.array([np.mean(segments.loc[segments[cluster_column] == label, columns_mean], axis=0) for label in cluster_labels])
   
    # Classify clusters
    if image_mode == "single":
        flood_clusters = []
        fv_clusters = []
        pw_clusters = cluster_labels[(cluster_centroids[:,columns_mean.index(flood_vh_label)] < t_vh) & (cluster_centroids[:,columns_mean.index(flood_vv_label)] < t_vv)]
    else:
        flood_clusters = cluster_labels[(cluster_centroids[:,columns_mean.index(flood_vh_label)] < t_vh) \
                                        & (cluster_centroids[:,columns_mean.index(flood_vv_label)] < t_vv)]
        pw_clusters = cluster_labels[(cluster_centroids[:,columns_mean.index(flood_vh_label)] < t_vh) & (cluster_centroids[:,columns_mean.index(flood_vv_label)] < t_vv) \
                               & (cluster_centroids[:,columns_mean.index(ref_vh_label)] < t_vh) & (cluster_centroids[:,columns_mean.index(ref_vv_label)] < t_vv)]
        fv_clusters = cluster_labels[(cluster_centroids[:,columns_mean.index(inc_vv_label)] >= t_incvv) & (cluster_centroids[:,columns_mean.index(inc_r_label)] >= t_incr)]

    # Relabel classified clusters
    y_cluster_classes = np.zeros(segments[cluster_column].shape)
    for clusterlabel in fv_clusters:
        y_cluster_classes[segments[cluster_column] == clusterlabel] = 3
    for clusterlabel in flood_clusters:
        y_cluster_classes[segments[cluster_column] == clusterlabel] = 2
    for clusterlabel in pw_clusters:
        y_cluster_classes[segments[cluster_column] == clusterlabel] = 1

    # Save output
    if output_filename:
        print("{} - Saving cluster classif. output to {}...".format(datetime.datetime.now(), output_filename))
        with open(output_filename, "wb") as handle:
            pickle.dump(y_cluster_classes, handle)

    # Plot data with centroids
    if fig_filename:
        print("{} - Plotting clusters and saving to {}...".format(datetime.datetime.now(), fig_filename))
        colorlist = ['coral', 'lightblue', 'blue', 'lightgreen', 'green', 'grey', 'purple',
                      'yellow', 'red', 'pink', 'saddlebrown', 'cyan', 'violet', 'olive']
        segments["clusclass"] = y_cluster_classes
        fig, ax = pf.makeScatterSubplotsObjects(segments, columns_mean, colorlabel="clusclass", cmap=mpl.colors.ListedColormap(colorlist[:np.max(y_cluster_classes)+1]))
        fig, ax = pf.plotCentroids(cluster_centroids, columns_mean, columns_mean, fig=fig, ax=ax, colors=colorlist[:num_clusters])
        ax = pf.plotThresholds(t_vv, t_vh, columns_mean, ax)
        fig.set_size_inches(12, 8)
        plt.savefig(fig_filename)
        plt.close(fig)
        
    return y_cluster_classes


def apply_spectralclustering(segments, n_clusters, feature_space, random_state=0, spec_affinity="nearest_neighbors", directory_output=None, segments_name="Segments", return_labels=False):
    """
    Apply spectral clustering on segments based on feature_space
    
    Inputs:
    segments: geopandas GeoDataFrame
        Objects to consider for clustering.
    n_clusters: int
        Number of clusters.
    feature_space: list or str
        List of band names to include in FS or keyword describing FS. Keyword should be constructed
        as "bands_stats", where bands can be one or a combination of "SARflood", "SAR", "wR", "wC",
        "incF", "wHAND", "o3", "opt", "lc" or "lc_frac" and stats should be "mean" or "all" (with "all"
        referring to mean + std).
    random_state: int or None (default=0)
        Determines random number generation for centroid initialization. Use an int to make the randomness deterministic.
    spec_affinity: str or callable (default='nearest_neighbors')
        How to construct the affinity matrix (see sklearn.cluster.SpectralClustering docs).
    directory_output: str or None (default=None)
        If not None, clustering output (+ comp. time) and scatter plot will be saved to this directory.
    segments_name: str (default='Segments')
        Prefix to use when saving output.
    return_labels: bool (default=False)
        Whether to return the labels of the feature space.
    Outputs:
    segments_cluster: geopandas Series
        Cluster labels of all objects.
    labels (if return_labels==True): list
        List of feature (band) labels.
    """
    X, labels = prepare_data(segments, feature_space)
    if X is None:
        return None

    ts = datetime.datetime.now()
    cluster = cl.SpectralClustering(n_clusters=n_clusters, affinity=spec_affinity)
    cluster = cluster.fit(X)
    segments["cluster"] = cluster.labels_
    te = datetime.datetime.now()

    if directory_output:
        outfile_pickle = os.path.join(directory_output, "{}_SpecClust_{}_{}cl.pkl".format(segments_name, feature_space, n_clusters))
        print("{} - Saving cluster output to {}...".format(datetime.datetime.now(), outfile_pickle))
        with open(outfile_pickle, "wb") as handle:
            pickle.dump([segments["cluster"], (te-ts).total_seconds()], handle)

        outfile_fig = os.path.join(directory_output, "{}_SpecClust_{}_{}cl.png".format(segments_name, feature_space, n_clusters))
        print("{} - Plotting clusters and saving to {}...".format(datetime.datetime.now(), outfile_fig))
        colors = ['coral', 'lightblue', 'blue', 'lightgreen', 'green', 'grey', 'purple',
                      'yellow', 'red', 'pink', 'saddlebrown', 'cyan', 'violet', 'olive']
        labels_mean = [el for el in labels if el.endswith("mean") and ("VV" in el or "VH" in el)]
        fig, ax = pf.makeScatterSubplotsObjects(segments, labels_mean, colorlabel="cluster", cmap=mpl.colors.ListedColormap(colors[:n_clusters]))
        if fig:
            fig.suptitle('Object intensity means per cluster', fontsize=16)
            plt.savefig(outfile_fig)

    if return_labels:
        return segments["cluster"], labels
    return segments["cluster"]


def apply_quickshiftclustering(segments, feature_space, bandwidth_density=0.2, bandwidth_nn=1.0, min_cluster_size=100, num_clusters=None, directory_output=None, segments_name="Segments", return_labels=False):
    """ Apply quickshift clustering on segments based on feature_space
    
    Inputs:
    segments: geopandas GeoDataFrame
        Objects to consider for clustering.
    n_clusters: int
        Number of clusters.
    feature_space: list or str
        List of band names to include in FS or keyword describing FS. Keyword should be constructed
        as "bands_stats", where bands can be one or a combination of "SARflood", "SAR", "wR", "wC",
        "incF", "wHAND", "o3", "opt", "lc" or "lc_frac" and stats should be "mean" or "all" (with "all"
        referring to mean + std).
    bandwidth_density: float (default=0.2)
        Bandwidth for density calculation.
    bandwidth_nn: float (default=1.0)
        Bandwidth for nearest neighbour assignment.
    min_cluster_size: int or None(default=100)
        Min. cluster size to be considered as 'real' cluster.
    num_clusters: int or None (default=None)
        Number of (largest) clusters to consider as 'real' cluster.
    directory_output: str or None (default=None)
        If not None, clustering output (+ comp. time) and scatter plot will be saved to this directory.
    segments_name: str (default='Segments')
        Prefix to use when saving output.
    return_labels: bool (default=False)
        Whether to return the labels of the feature space.
    Outputs:
    segments_cluster: geopandas Series
        Cluster labels of all objects.
    labels (if return_labels==True): list
        List of feature (band) labels.
   """
    X, labels = prepare_data(segments, feature_space)
    if X is None:
        return None
    
    print("{} - Calculating density...".format(datetime.datetime.now()))
    nbrs = NearestNeighbors(radius=bandwidth_density).fit(X)
    density = []
    for el in X:
        density.append(get_point_density(el, X, nbrs))
    density = np.array(density)
    segments["logdensity"] = np.log10(density)
    
    print("{} - Calculating cluster seeds...".format(datetime.datetime.now()))
    nbrs = NearestNeighbors(radius=bandwidth_nn).fit(X)
    parent = []
    for i, point, point_density in zip(np.arange(X.shape[0]), X, density):
        dist, ind = nbrs.radius_neighbors([point]) # find neighbors within bandwith_nn of point
        ind = ind[np.argsort(dist)] # sorting neighbors according to distance
        ind = ind[0]
        ind = ind[density[ind] > point_density] # keep only neighbors with a higher density
        if ind.size != 0:
            parent.append(ind[0]) # point gets assigned to cluster of nearest neighbor with higher density
        else:
            parent.append(i) # point becomes cluster seed if no neighbors with higher density
    parent = np.array(parent)
    
    print("{} - Flattening forest of parent points...".format(datetime.datetime.now()))
    old = np.zeros_like(parent)
    while (old != parent).any():
        old = parent
        parent = parent[parent]
        
    print("{} cluster seeds found.".format(np.unique(parent).size))
    segments["cluster"] = np.unique(parent, return_inverse=True)[1] # relabel clusters from 0:n
    
    if min_cluster_size:
        print("{} - Removing outlier clusters...".format(datetime.datetime.now()))
        unique_clusters, unique_cluster_counts = np.unique(segments["cluster"], return_counts=True)
        freq_all = np.array([unique_cluster_counts[i] for i in segments["cluster"]])
        segments.loc[freq_all < min_cluster_size, ["cluster"]] = -1
        segments["cluster"] = np.unique(segments["cluster"], return_inverse=True)[1] # relabel clusters from 0:n    
        num_realclusters = np.unique(segments["cluster"]).size - 1
        num_outlierclusters = unique_clusters.size - num_realclusters
        print("{} real cluster seeds and {} outlier cluster seeds found.".format(num_realclusters, num_outlierclusters))
    elif num_clusters: # TODO: verify
        print("{} - Selecting only the {} largest clusters...".format(datetime.datetime.now(), num_clusters))
        unique_clusters, unique_cluster_counts = np.unique(segments["cluster"], return_counts=True)
        freq_all = np.array([unique_cluster_counts[i] for i in segments["cluster"]])
        unique_cluster_counts_sorted = np.sort(unique_cluster_counts) # ascending
        t_size = unique_cluster_counts_sorted[-num_clusters]
        segments.loc[freq_all < t_size, ["cluster"]] = -1
        segments["cluster"] = np.unique(segments["cluster"], return_inverse=True)[1] # relabel clusters from 0:n    
        num_realclusters = np.unique(segments["cluster"]).size - 1
        num_outlierclusters = unique_clusters.size - num_realclusters
        print("{} real cluster seeds and {} outlier cluster seeds found.".format(num_realclusters, num_outlierclusters))
    else:
        print("Error! At least one of [min_cluster_size, num_clusters], should be specified. Aborting.")
        if return_labels:
            return None, None
        return None

    if directory_output:
        outfile_pickle = os.path.join(directory_output, "{}_QuickClust_{}_bd{}_bn{}_ms{}.pkl".format(segments_name, feature_space, bandwidth_density, bandwidth_nn, min_cluster_size))
        print("{} - Saving cluster output to {}...".format(datetime.datetime.now(), outfile_pickle))
        with open(outfile_pickle, "wb") as handle:
            pickle.dump(segments["cluster"], handle)

        colors = ['coral', 'lightblue', 'blue', 'lightgreen', 'green', 'grey', 'purple',
                  'yellow', 'red', 'pink', 'saddlebrown', 'cyan', 'violet', 'olive']
        outfile_fig = os.path.join(directory_output, "QuickshiftClustering_{}_bd{}_bn{}_ms{}.png".format(feature_space, bandwidth_density, bandwidth_nn, min_cluster_size))
        print("{} - Plotting clusters and saving to {}...".format(datetime.datetime.now(), outfile_fig))
        labels_mean = [el for el in labels if el.endswith("mean") and ("VV" in el or "VH" in el)]
        fig, ax = pf.makeScatterSubplotsObjects(segments, labels_mean, colorlabel="cluster", cmap=mpl.colors.ListedColormap(colors[:num_realclusters+1]))
        fig.set_size_inches(12,7)
        fig.suptitle('Object intensity means colored by cluster', fontsize=16)
        plt.savefig(outfile_fig)
        plt.close(fig)
        outfile_fig = os.path.join(directory_output, "QuickshiftClustering_{}_bd{}_bn{}_ms{}_density.png".format(feature_space, bandwidth_density, bandwidth_nn, min_cluster_size))        
        print("{} - Plotting segments density to {}...".format(datetime.datetime.now(), outfile_fig))
        fig, ax = pf.makeScatterSubplotsObjects(segments, labels_mean, colorlabel="logdensity", cmap='jet')
        fig.set_size_inches(12,7)
        fig.suptitle('Object intensity means colored by density', fontsize=16)
        plt.savefig(outfile_fig)      
        plt.close(fig)
        outfile_fig = os.path.join(directory_output, "QuickshiftClustering_{}_bd{}_bn{}_ms{}_spatial.png".format(feature_space, bandwidth_density, bandwidth_nn, min_cluster_size))        
        print("{} - Plotting clusters spatially to {}...".format(datetime.datetime.now(), outfile_fig))
        fig, ax = plt.subplots()
        segments.plot(column="cluster", cmap=mpl.colors.ListedColormap(colors[:len(np.unique(segments["cluster"]))]), ax=ax)
        fig.set_size_inches(13,10)
        plt.savefig(outfile_fig)
        plt.close(fig)

    if return_labels:
        return segments["cluster"], labels
    return segments["cluster"]


def classify_quickshiftclustering(segments, cluster_column, t_vv, t_vh, t_incvv, t_incr, outlier_classification="neighbor", image_mode="dual", output_filename=None, fig_filename=None):
    """
    Classify quickshift clusters
    
    
    Inputs:
    segments: geopandas GeoDataFrame
        Objects to consider for clustering.
    cluster_column: str
        Label of column (added to gdf) containing cluster classification.
    t_vv: float
        VV threshold for classification of PW/OF.
    t_vh: float
        VH threshold for classification of PW/OF.
    t_incvv: float (default=1)
        incVV threshold for classification of FV.
    t_incr: float (default=1)
        incR threshold for classification of FV.
    image_mode: 'single', or 'dual' (default='dual')
        Label indicated whether a single or two scenes are considered for classification.
    output_filename: str or None (default=None)
        If not None, path to save cluster classification output (as pickle).
    fig_filename: str or None (default=None)
        If not None, path to save scatter plot colored by cluster class.
    Outputs:
    segments_cluster_class: geopandas Series
        Cluster classification labels of all objects (0=DL, 1=PW, 2=OF, 3=FV).
    """
    # Gather features used for classification
    flood_vh_label = "fVH_mean"
    flood_vv_label = "fVV_mean"
    ref_vh_label = "rVH_mean"
    ref_vv_label = "rVV_mean"
    inc_vv_label = "incVV_mean"
    inc_r_label = "incR_mean"
    columns_mean = [flood_vh_label, flood_vv_label, ref_vh_label, ref_vv_label, inc_vv_label, inc_r_label]

    # Reclassify outlier points
    num_outliers = np.sum(segments[cluster_column] == 0)
    if num_outliers > 1 and num_outliers < len(segments):
        # Assign outlier clusters to nearest neighbor
        print("{} - Reclassifying {} segments of outlier cluster".format(datetime.datetime.now(), num_outliers))
        outlier_ind = np.where(segments[cluster_column] == 0)[0]
        labeled_ind = np.where(segments[cluster_column] != 0)[0]
        if outlier_classification == "neighbor": # assign points to nearest labeled neighbor
            nn, ndist = sklearn.metrics.pairwise_distances_argmin_min(segments.loc[outlier_ind, columns_mean], segments.loc[labeled_ind, columns_mean], axis=1, metric='euclidean')
            segments.loc[outlier_ind, cluster_column] = np.array(segments.loc[labeled_ind[nn], cluster_column])            
        elif "centroid" in outlier_classification:
            # Calculate cluster centroids
            cluster_labels = np.unique(segments.loc[segments[cluster_column] != 0, cluster_column])
            cluster_centroids = np.array([np.mean(segments.loc[segments[cluster_column] == label, columns_mean], axis=0) for label in cluster_labels])
            # Assign points to nearest centroid
            nn, ndist = sklearn.metrics.pairwise_distances_argmin_min(segments.loc[outlier_ind, columns_mean], cluster_centroids, axis=1, metric='euclidean')
            segments.loc[outlier_ind, "cluster"] = np.array(segments.loc[labeled_ind[nn], "cluster"])
        else:
            print("Error! Non-existent option for outlier classification.")
    
    # Calculate cluster centroids
    cluster_labels = np.unique(segments[cluster_column])
    num_clusters = len(np.unique(cluster_labels))
    cluster_centroids = np.array([np.mean(segments.loc[segments[cluster_column] == label, columns_mean], axis=0) for label in cluster_labels])

    # Classify clusters
    if image_mode == "single":
        flood_clusters = []
        fv_clusters = []
        pw_clusters = cluster_labels[(cluster_centroids[:,columns_mean.index(flood_vh_label)] < t_vh) & (cluster_centroids[:,columns_mean.index(flood_vv_label)] < t_vv)]
    else:
        flood_clusters = cluster_labels[(cluster_centroids[:,columns_mean.index(flood_vh_label)] < t_vh) \
                                        & (cluster_centroids[:,columns_mean.index(flood_vv_label)] < t_vv)]
        pw_clusters = cluster_labels[(cluster_centroids[:,columns_mean.index(flood_vh_label)] < t_vh) & (cluster_centroids[:,columns_mean.index(flood_vv_label)] < t_vv) \
                               & (cluster_centroids[:,columns_mean.index(ref_vh_label)] < t_vh) & (cluster_centroids[:,columns_mean.index(ref_vv_label)] < t_vv)]
        fv_clusters = cluster_labels[(cluster_centroids[:,columns_mean.index(inc_vv_label)] >= t_incvv) & (cluster_centroids[:,columns_mean.index(inc_r_label)] >= t_incr)]

    # Relabel classified clusters
    y_cluster_classes = np.zeros(segments[cluster_column].shape)
    for clusterlabel in fv_clusters:
        y_cluster_classes[segments[cluster_column] == clusterlabel] = 3
    for clusterlabel in flood_clusters:
        y_cluster_classes[segments[cluster_column] == clusterlabel] = 2
    for clusterlabel in pw_clusters:
        y_cluster_classes[segments[cluster_column] == clusterlabel] = 1
    
    # Save output
    if output_filename:
        print("{} - Saving cluster classif. output and no-outlier segments to {}...".format(datetime.datetime.now(), output_filename))
        with open(output_filename, "wb") as handle:
            pickle.dump([y_cluster_classes, segments["cluster"]], handle)

    # Plot data with centroids
    if fig_filename:
        print("{} - Plotting clusters and saving to {}...".format(datetime.datetime.now(), fig_filename))
        colorlist = ['coral', 'lightblue', 'blue', 'lightgreen', 'green', 'grey', 'purple',
                      'yellow', 'red', 'pink', 'saddlebrown', 'cyan', 'violet', 'olive']
        segments["clusclass"] = y_cluster_classes
        fig, ax = pf.makeScatterSubplotsObjects(segments, columns_mean, colorlabel="clusclass", cmap=mpl.colors.ListedColormap(colorlist[:np.max(y_cluster_classes)+1]))
        fig, ax = pf.plotCentroids(cluster_centroids, columns_mean, columns_mean, fig=fig, ax=ax, colors=colorlist[:num_clusters])
        ax = pf.plotThresholds(t_vv, t_vh, columns_mean, ax)
        fig.set_size_inches(12, 8)
        plt.savefig(fig_filename)
        plt.close(fig)
        
    return y_cluster_classes

