# -*- coding: utf-8 -*-
"""
@author: Lisa Landuyt
@purpose: Ancillary functions for accuracy assessment
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn

def scatterplot_objects(segments, labels, colorlabel=None, cmap=None, fig_filename=None, return_mappable=False, axmin=None, axmax=None, vmin=None, vmax=None, alpha=1, normed=False):
    """ Make scatter plots of segments for label combinations
    
    Inputs:
    segments: geopandas GeoDataFrame
        Segments to plot.
    labels: list
        Labels of columns to plot. Should be of length 2/3/4.
        If len(labels) == 2, 1 subplot is created with x=label[0] and y=label[1].
        If len(labels) == 3, 1x3 subplots are created with (x,y) equal to
        (label[0],label[1]), (label[0],label[2]), (label[1],label[2]).
        If len(labels) == 4, 2x2 subplot are created with (x,y) equal to
        (label[0],label[1]), (label[2],label[3]), (label[0],label[2]), (label[1],label[3]).
    colorlabel: str or None (default=None)
        Label of column to use for coloring.
    cmap: matplotlib colormap instance or None (default=None)
        Colormap.
    fig_filename: str or None (default=None)
        If not None, figure is saved to specified path.
    return_mappable: bool (default=False)
        If true, mappable is returned.
    axmin: float or None (default=None)
        Min. limit for x and y axis.
    axmax: float or None (default=None)
        Max. limit for x and y axis.
    vmin: float or None (default=None)
        Min. value for color axis.
    vmax: float or None (default=None)
        Max. value for color axis.
    alpha: float (default=1)
        Transparency of plotted points.
    normed: bool (default=False)
        If True, data are normalized to 0 mean and unit variance.
    Outputs:
    fig, ax: tuple
        Handles to figure and axis.
    sc: matplotlib PathCollection
        Handle to mappable.
    """
    colors = ['coral', 'lightblue', 'blue', 'lightgreen', 'green', 'grey', 'purple',
                  'yellow', 'red', 'pink', 'saddlebrown', 'cyan', 'violet', 'olive']
    if colorlabel is None:
        color = 'b'
        cmap = None
    else:
        color = segments[colorlabel]
        if cmap is None:
            if np.unique(segments[colorlabel]).size > 1000:
                cmap = 'jet'
            elif np.unique(segments[colorlabel]).size <= len(colors):
                cmap = mpl.colors.ListedColormap(colors[:np.unique(segments[colorlabel]).size])
            else:
                cmap = mpl.colors.ListedColormap(np.random.rand(np.unique(segments[colorlabel]).size,3))
    if normed:
            segments[labels] = sklearn.preprocessing.scale(segments[labels])
    if len(labels) == 4:
        fig, ax = plt.subplots(2, 2)
        ax[0,0].scatter(segments[labels[0]], segments[labels[1]], c=color, s=0.5, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
        ax[0,1].scatter(segments[labels[2]], segments[labels[3]], c=color, s=0.5, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
        ax[1,0].scatter(segments[labels[0]], segments[labels[2]], c=color, s=0.5, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
        sc = ax[1,1].scatter(segments[labels[1]], segments[labels[3]], c=color, s=0.5, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
        ax[0,0].set_xlabel(labels[0])
        ax[0,0].set_ylabel(labels[1])
        ax[0,1].set_xlabel(labels[2])
        ax[0,1].set_ylabel(labels[3])
        ax[1,0].set_xlabel(labels[0])
        ax[1,0].set_ylabel(labels[2])
        ax[1,1].set_xlabel(labels[1])
        ax[1,1].set_ylabel(labels[3])
        if axmin is None:
            axmin = min([min([subax.get_xlim()[0] for subax in ax.ravel()]), min([subax.get_ylim()[0] for subax in ax.ravel()])])
        if axmax is None:
            axmax = max([max([subax.get_xlim()[1] for subax in ax.ravel()]), max([subax.get_ylim()[1] for subax in ax.ravel()])])
        for subax in ax.ravel():
            subax.set_xlim(axmin, axmax)
            subax.set_ylim(axmin, axmax)
    elif len(labels) == 2:
        fig, ax = plt.subplots()
        ax.scatter(segments[labels[0]], segments[labels[1]], c=color, s=0.5, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        if axmin is None:
            axmin = min([ax.get_xlim()[0] , ax.get_ylim()[0]])
        if axmax is None:
            axmax = max([ax.get_xlim()[1] , ax.get_ylim()[1]])
        ax.set_xlim(axmin, axmax)
        ax.set_ylim(axmin, axmax)
    elif len(labels) == 3: 
        fig, ax = plt.subplots(2, 2)
        ax[0,0].scatter(segments[labels[0]], segments[labels[1]], c=color, s=0.5, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
        ax[0,1].scatter(segments[labels[0]], segments[labels[2]], c=color, s=0.5, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
        sc = ax[1,0].scatter(segments[labels[1]], segments[labels[2]], c=color, s=0.5, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
        ax[0,0].set_xlabel(labels[0])
        ax[0,0].set_ylabel(labels[1])
        ax[0,1].set_xlabel(labels[0])
        ax[0,1].set_ylabel(labels[2])
        ax[1,0].set_xlabel(labels[1])
        ax[1,0].set_ylabel(labels[2])
        ax[1,1].axis('off')
    else:
        print("Number of labels should equal 2, 3 or 4. Aborting.")
        fig, ax, sc = (None, None, None)
    if fig_filename:
        plt.savefig(fig_filename)
    if return_mappable:
        return fig, ax, sc
    else:
        return fig, ax


def scatterplot_pixels(image, image_bandnames, labels, colorarray=None, cmap=None, fig_filename=None):
    """ Make scatter plots of segments for label combinations
    
    Inputs:
    image: ndarray
        Pixels to plot.
    image_bandnames: list
        Labels corresponding to image bands.
    labels: list
        Labels of columns to plot. Should be of length 4.
    colorarray: ndarray
        Values of color corresponding to each pixel.
    cmap: matplotlib colormap instance or None (default=None)
        Colormap.
    fig_filename: str or None (default=None)
        If not None, figure is saved to specified path.
    Outputs:
    fig, ax: tuple
        Handles to figure and axis.
    sc: matplotlib PathCollection
        Handle to mappable.
    """
    if colorarray is None:
        color = None
        cmap = None
    else:
        color = colorarray
        if cmap is None:
            cmap = mpl.colors.ListedColormap(np.random.rand(np.unique(colorarray).size,3))
    label_indices = []
    for label in labels:
        label_indices.append(image_bandnames.index(label))
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    ax[0,0].scatter(image[label_indices[0]], image[label_indices[1]], c=color, s=0.5, cmap=cmap)
    ax[0,1].scatter(image[label_indices[2]], image[label_indices[3]], c=color, s=0.5, cmap=cmap)
    ax[1,0].scatter(image[label_indices[0]], image[label_indices[2]], c=color, s=0.5, cmap=cmap)
    ax[1,1].scatter(image[label_indices[1]], image[label_indices[3]], c=color, s=0.5, cmap=cmap)
    ax[0,0].set_xlabel(labels[0])
    ax[0,0].set_ylabel(labels[1])
    ax[0,1].set_xlabel(labels[2])
    ax[0,1].set_ylabel(labels[3])
    ax[1,0].set_xlabel(labels[0])
    ax[1,0].set_ylabel(labels[2])
    ax[1,1].set_xlabel(labels[1])
    ax[1,1].set_ylabel(labels[3])
    if fig_filename:
        plt.savefig(fig_filename)
    return fig, ax


def plot_centroids(cluster_centroids, labels, band_names, fig=None, ax=None, colors=None):
    """Plot cluster centroids on scatter plot
    
    Inputs:
    cluster_centroids: ndarray
        Array of cluster centroids (n_rows = n_clusters, n_cols=n_dim_fs).
    labels: list
        Labels of columns to plot. See doc of scatterplot_objects.
    band_names: list
        Band names corresponding to 2nd dimension cluster_centroids.
    fig: handle or None (default=None)
        Figure handle.
    ax: handle or None (default=None)
        Axis handle.
    colors: list or None (default=None)
        List of colors to plot centroids in.
    Outputs:
    fig, ax: tuple
        Handles to figure and axis.
    """
    s = 36
    marker = "o"
    if ax is None or fig is None:
        if len(labels) == 2:
            fig, ax = plt.subplots()
        elif len(labels) == 4:
            fig, ax = plt.subplots(2,2)
    if colors is None:
        colors = 'black'
    elif len(colors) != cluster_centroids.shape[0]:
        print("Length of colors does not match number of centroids. Centroids will be plotted in black.")
        colors = 'black'
    if len(labels) == 2:
        ax.scatter(cluster_centroids[:,band_names.index(labels[0])], cluster_centroids[:,band_names.index(labels[1])], marker=marker, s=s, c=colors, edgecolor='black')
    elif len(labels) == 4:
        ax[0,0].scatter(cluster_centroids[:,band_names.index(labels[0])], cluster_centroids[:,band_names.index(labels[1])], s=s, marker=marker, c=colors, edgecolor='black')
        ax[0,1].scatter(cluster_centroids[:,band_names.index(labels[2])], cluster_centroids[:,band_names.index(labels[3])], s=s, marker=marker, c=colors, edgecolor='black')
        ax[1,0].scatter(cluster_centroids[:,band_names.index(labels[0])], cluster_centroids[:,band_names.index(labels[2])], s=s, marker=marker, c=colors, edgecolor='black')
        ax[1,1].scatter(cluster_centroids[:,band_names.index(labels[1])], cluster_centroids[:,band_names.index(labels[3])], s=s, marker=marker, c=colors, edgecolor='black')
    elif len(labels) == 3:
        ax[0,0].scatter(cluster_centroids[:,band_names.index(labels[0])], cluster_centroids[:,band_names.index(labels[1])], s=s, marker=marker, c=colors, edgecolor='black')
        ax[0,1].scatter(cluster_centroids[:,band_names.index(labels[0])], cluster_centroids[:,band_names.index(labels[2])], s=s, marker=marker, c=colors, edgecolor='black')
        ax[1,0].scatter(cluster_centroids[:,band_names.index(labels[1])], cluster_centroids[:,band_names.index(labels[2])], s=s, marker=marker, c=colors, edgecolor='black')
    return fig, ax


def plot_thresholds(t_vv, t_vh, labels, ax):
    """Plot thresholds on scatter plot
    
    Inputs:
    t_vv: float
        VV threshold.
    t_vh: float
        VH threshold.
    labels: list
        Labels of columns to plot. See doc of scatterplot_objects.
        If len(labels) == 2, it is assumed label 0 is VV and label 1 is VH.
        If len(labels) == 3, it is assumed label 0 is VV and label 1 is VH.
        If len(labels) == 4, it is assumed label 0/2 is VV and label 1/3 is VH.
    ax: handle
        Axis handle.
    Ouputs:
    ax: handle
        Axis handle.
    """
    if len(labels) == 2:
        minx, maxx = ax.get_xlim()
        miny, maxy = ax.get_ylim()
        ax.plot([minx, t_vv], [t_vh, t_vh], linestyle='dashed', color='gray') 
        ax.plot([t_vv, t_vv], [miny, t_vh], linestyle='dashed', color='gray') 
    elif len(labels) == 3:
        min0, max0 = ax[0].get_xlim()
        min1, max1 = ax[0].get_ylim()
        min2, max2 = ax[1].get_ylim()
        ax[0].plot([min0, max0], [t_vh, t_vh], linestyle='dashed', color='gray') 
        ax[0].plot([t_vv, t_vv], [min1, max1], linestyle='dashed', color='gray') 
        ax[1].plot([t_vv, t_vv], [min2, max2], linestyle='dashed', color='gray') 
        ax[2].plot([t_vh, t_vh], [min2, max2], linestyle='dashed', color='gray') 
        ax[0].set_xlim(min0, max0)
        ax[0].set_ylim(min1, max1)
        ax[1].set_xlim(min0, max0)
        ax[1].set_ylim(min2, max2)
        ax[2].set_xlim(min1, max1)
        ax[2].set_ylim(min2, max2)
    elif len(labels) == 4:
        minx, maxx = ax[0,0].get_xlim()
        miny, maxy = ax[0,0].get_ylim()
        ax[0,0].plot([minx, maxx], [t_vh, t_vh], linestyle='dashed', color='gray') 
        ax[0,0].plot([t_vv, t_vv], [miny, maxy], linestyle='dashed', color='gray') 
        ax[0,1].plot([minx, maxx], [t_vh, t_vh], linestyle='dashed', color='gray') 
        ax[0,1].plot([t_vv, t_vv], [miny, maxy], linestyle='dashed', color='gray') 
        ax[1,0].plot([minx, maxx], [t_vv, t_vv], linestyle='dashed', color='gray') 
        ax[1,0].plot([t_vv, t_vv], [miny, maxy], linestyle='dashed', color='gray') 
        ax[1,1].plot([minx, maxx], [t_vh, t_vh], linestyle='dashed', color='gray') 
        ax[1,1].plot([t_vh, t_vh], [miny, maxy], linestyle='dashed', color='gray') 
    return ax
    