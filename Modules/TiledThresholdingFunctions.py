# -*- coding: utf-8 -*-
"""
@author: Lisa Landuyt
@purpose: Ancillary functions for threshold selection
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle


def apply_ki(image, accuracy=200, plot_j=False): # TODO: save fig to file?
    """Select threshold according to Kittler & Illingworth
    
    Inputs:
    image: nd array
        Array of pixel values.
    accuracy: int(default=200)
        Number of bins to construct histogram.
    plot_j: bool (default=False)
        Whether to plot the cost function
    Outputs:
    t: float
        Threshold value
    """
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)
    # Histogram    
    h, bin_edges = np.histogram(image[~np.isnan(image)], bins=accuracy, density=True)
    bin_width = bin_edges[1]-bin_edges[0]
    g = np.arange(bin_edges[0]+bin_width/2.0, bin_edges[-1], bin_width)
    g_pos = g - np.min(g);
    g01 = g_pos / np.max(g_pos);
    
    # Cost function and threshold
    c = np.cumsum(h)
    m = np.cumsum(h * g01)
    s = np.cumsum(h * g01**2)
    cb = c[-1] - c
    mb = m[-1] - m
    sb = s[-1] - s
    c[c == 0] = 1e-9
    cb[cb == 0] = 1e-9
    var_f = s/c - (m/c)**2
    if np.any(var_f < 0):
        var_f[var_f < 0] = 0
    sigma_f = np.sqrt(var_f)
    var_b = sb/cb - (mb/cb)**2
    if np.any(var_b < 0):
        var_b[var_b < 0] = 0
    sigma_b = np.sqrt(var_b)
    p = c / c[-1]
    sigma_f[sigma_f == 0] = 1e-9
    sigma_b[sigma_b == 0] = 1e-9
    j = p * np.log(sigma_f) + (1-p)*np.log(sigma_b) - p*np.log(p) - (1-p)*np.log(1-p+1e-9)
    j[~np.isfinite(j)] = np.nan
    idx = np.nanargmin(j)
    t = g[idx]
    # Plot
    if plot_j:
        fig, ax = plt.subplots(2,1)
        ax[0].plot(g, j, color='k')
        ax[0].plot([t, t], [np.nanmin(j), np.nanmax(j)], 'r')
        ax[1].bar(g, h)
        ax[1].plot([t, t], [0, np.nanmax(h)], 'r')
    # Return
    return t

def apply_otsu(image, accuracy=200, plot_j=False):
    """Select threshold according to Otsu
    
    Inputs:
    image: nd array
        Array of pixel values.
    accuracy: int(default=200)
        Number of bins to construct histogram.
    plot_j: bool (default=False)
        Whether to plot the cost function
    Outputs:
    t: float
        Threshold value
    """
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)
    # Histogram
    h, bin_edges = np.histogram(image[~np.isnan(image)], bins=accuracy, density=True)
    bin_width = bin_edges[1]-bin_edges[0]
    g = np.arange(bin_edges[0]+bin_width/2.0, bin_edges[-1], bin_width)
    # Between class variance and threshold
    w1 = np.cumsum(h)
    w2 = w1[-1] - w1
    w2[w2 == 0] = 1e-9
    gh = np.cumsum(g*h)
    mu1 = gh/w1
    mu2 = (gh[-1]-gh)/w2
    var_between = w1*w2*(mu1-mu2)**2
    idx = np.nanargmax(var_between)
    t = g[idx]
    # Plot
    if plot_j:
        fig, ax = plt.subplots(2,1)
        ax[0].plot(g, var_between, color='k')
        ax[0].plot([t, t], [ax[0].get_ylim()[0], np.nanmax(var_between)], 'r')
        ax[1].bar(g, h)
        ax[1].plot([t, t], [0, np.nanmax(h)], 'r')
    # Return
    return t
        
def tile_vars(image, selection='Martinis', t_method=['KI', 'Otsu'], tile_dim=[200, 200], hand_matrix=None, incomplete_tile_warning=True):
    """ Calculate tile variables
    
    Inputs:
    image: nd array
        Array of pixel values.
    selection: str (default='Martinis')
        Method for tile selection. Currently only option is 'Martinis'.
    t_method: list (default=['KI', 'Otsu'])
        List of thresholds to calculate. Should contain one or both of "KI", "Otsu".
    tile_dim: list (default=[200, 200])
        Dimension of tiles.
    hand_matrix: ndarray or None (default=None)
        Array of HAND values.
    incomplete_tile_warning: bool (default=True)
        Whether to give a warning when incomplete tiles are encountered.
    Outputs:
    tile_ki: nd array
        Array of KI thresholds.
    tile_o: nd array
        Array of Otsu thresholds.
    average: nd array
        Array of tile averages.
    stdev: nd array
        Array of tiled st. devs.
    hand: nd array
        Array of tile mean HAND values.
    """
    tile_rows, tile_cols = tile_dim
    nrt = np.ceil(image.shape[0]/tile_rows).astype('int')
    nct = np.ceil(image.shape[1]/tile_cols).astype('int')
    if 'KI' in t_method:
        tile_ki = np.full([nrt, nct], np.nan)
    if 'Otsu' in t_method:
        tileO = np.full([nrt, nct], np.nan)
    if selection == 'Martinis':
        stdev = np.full([nrt, nct], np.nan)
        average = np.full([nrt, nct], np.nan)
    if hand_matrix is not None:
        hand = np.full([nrt, nct], np.nan)
    for r in np.arange(0, image.shape[0], tile_rows):
        tile_rindex = np.floor(r/tile_rows).astype('int')
        for c in np.arange(0, image.shape[1], tile_cols):
            tile_cindex = np.floor(c/tile_cols).astype('int')
            tile = image[r:min(r+tile_rows, image.shape[0]), c:min(c+tile_cols, image.shape[1])]
            if np.sum(np.isnan(tile)) <= 0.1*np.size(tile):
                if 'KI' in t_method:
                    tile_ki[tile_rindex, tile_cindex] = apply_ki(tile, 200)
                if 'Otsu' in t_method:
                    tileO[tile_rindex, tile_cindex] = apply_otsu(tile, 200)
                if selection == 'Martinis':
                    tr, tc = tile.shape
                    mu1 = np.nanmean(tile[0:tr//2, 0:tc//2])
                    mu2 = np.nanmean(tile[0:tr//2, tc//2:])
                    mu3 = np.nanmean(tile[tr//2:, 0:tc//2])
                    mu4 = np.nanmean(tile[tr//2:, tc//2:])
                    stdev[tile_rindex, tile_cindex] = np.std([mu1, mu2, mu3, mu4]) 
                    average[tile_rindex, tile_cindex] = np.nanmean(tile)
                if hand_matrix is not None:
                    hand[tile_rindex, tile_cindex] = np.nanmean(hand_matrix[r:min(r+tile_rows, image.shape[0]), c:min(c+tile_cols, image.shape[1])])
            elif incomplete_tile_warning:
                print("Tile ({0:.0f}, {1:.0f}) is incomplete.".format(tile_rindex, tile_cindex))
    if hand_matrix is not None:
        return tile_ki, tileO, average, stdev, hand  
    else:
        return tile_ki, tileO, average, stdev, None # Modify this line to include other selection methods!

def tiled_thresholding(image, selection='Martinis', t_method=['KI', 'Otsu'], tile_dim=[200, 200], n_final=5, hand_matrix=None, hand_t=100, directory_figure=None, incomplete_tile_warning=True):
    """ Apply tiled thresholding
    
    Inputs:
    image: nd array
        Array of pixel values.
    selection: str (default='Martinis')
        Method for tile selection. Currently only option is 'Martinis'.
    t_method: list (default=['KI', 'Otsu'])
        List of thresholds to calculate. Should contain one or both of "KI", "Otsu".
    tile_dim: list (default=[200, 200])
        Dimension of tiles.
    n_final: int (default=5)
        Number of tiles to select.
    hand_matrix: ndarray or None (default=None)
        Array of HAND values.
    hand_t: float (default=100)
        Maximum HAND value allowed for threshold selection.
    directory_figure: str or None (default=None)
        If not None, figure is saved to specified directory.
    incomplete_tile_warning: bool (default=True)
        Whether to give a warning when incomplete tiles are encountered.
    Outputs:
    tile_ki: nd array
        Array of KI thresholds.
    tile_o: nd array
        Array of Otsu thresholds.
    average: nd array
        Array of tile averages.
    stdev: nd array
        Array of tiled st. devs.
    hand: nd array
        Array of tile mean HAND values.
    """
    tile_dim = np.array(tile_dim)
    # Tile properties
    tile_ki, tileO, average, stdev, hand = tile_vars(image, selection=selection, t_method=t_method, tile_dim=tile_dim, hand_matrix=hand_matrix, incomplete_tile_warning=incomplete_tile_warning)
    # Tile selection
    q = np.nanpercentile(stdev, 95)
    stdev[average > np.nanmean(average)] = np.nan
    if hand_matrix:
        stdev[hand > 100] = np.nan
    i_r, i_c = np.where(stdev > q) # select tiles with stdev > 95-percentile
    while len(i_r) == 0:
        tile_dim = tile_dim//2
        print('Tile dimensions halved.')
        tile_ki, tileO, average, stdev, hand = tile_vars(image, selection=selection, t_method=t_method, tile_dim=tile_dim, hand_matrix=hand_matrix)
        q = np.percentile(stdev, 95)
        i_r, i_c = np.where(stdev > q)
    sorted_indices = np.argsort(stdev[i_r, i_c])[::-1]
    i_r = i_r[sorted_indices]
    i_c = i_c[sorted_indices]
    if i_r.size > n_final:
        tile_selection = [i_r[:n_final], i_c[:n_final]]
    else:
        tile_selection = [i_r, i_c]
    if directory_figure: 
        fig, ax = show_tileselection(image, tile_selection)
        plt.savefig(os.path.join(directory_figure, "Thresholding_TileSelection.png"))
    # Threshold and quality indicator
    if 'KI' in t_method:
        t_ki = np.mean(tile_ki[tuple(tile_selection)])
        s = np.std(tile_ki[tuple(tile_selection)])
        if s > 5:
            print('Histogram merge necessary for KI.')
            pixel_selection = np.empty(0)
            nrt, nct = tile_ki.shape
            for tile_rindex, tile_cindex in zip(tile_selection[0], tile_selection[1]):  
                tile = image[tile_rindex*tile_dim[0]:min((tile_rindex+1)*tile_dim[0], image.shape[0]), tile_cindex*tile_dim[1]:min((tile_cindex+1)*tile_dim[1], image.shape[1])]
                pixel_selection = np.append(pixel_selection, tile.ravel())
                del tile
            t_ki = apply_ki(pixel_selection)
    if 'Otsu' in t_method:
        t_otsu = np.mean(tileO[tuple(tile_selection)])
        s = np.std(tileO[tuple(tile_selection)])
        if s > 5:
            print('Histogram merge necessary for Otsu.')
            pixel_selection = np.empty(0)
            nrt, nct = tile_ki.shape
            for tile_rindex, tile_cindex in zip(tile_selection[0], tile_selection[1]): 
                tile = image[tile_rindex*tile_dim[0]:min((tile_rindex+1)*tile_dim[0], image.shape[0]), tile_cindex*tile_dim[1]:min((tile_cindex+1)*tile_dim[1], image.shape[1])]
                pixel_selection = np.append(pixel_selection, tile.ravel())
                del tile
            t_otsu = apply_otsu(pixel_selection)
    if 'KI' in t_method and 'Otsu' in t_method:
        return [t_ki, t_otsu], tile_selection
    elif 'KI' in t_method:
        return t_ki, tile_selection
    elif 'O' in t_method:
        return t_otsu, tile_selection
    
def show_tileselection(image, tile_selection, tile_dim=[200, 200]):
    """Plot image tiles and indicate selection
    
    Inputs:
    image: nd array
        Array of pixel values.
    tile_selection: list
        List of row and column indices selected tiles.
    tile_dim: list (default=[200, 200])
        Dimension of tiles.
    Outputs:
    fig: handle to figure
    ax: handle to axis
    """
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    for r in np.arange(image.shape[0]+1, step=200):
        ax.plot([0, image.shape[1]], [r, r], 'r')
    for c in np.arange(image.shape[1]+1, step=200):
        ax.plot([c, c], [0, image.shape[0]], 'r')    
    for tiler, tilec in zip(tile_selection[0], tile_selection[1]):
        ax.plot([tilec*tile_dim[0], tilec*tile_dim[0]], [tiler*tile_dim[0], (tiler+1)*tile_dim[0]], color=[0, 1, 0])
        ax.plot([(tilec+1)*tile_dim[0], (tilec+1)*tile_dim[0]], [tiler*tile_dim[0], (tiler+1)*tile_dim[0]], color=[0, 1, 0])
        ax.plot([tilec*tile_dim[0], (tilec+1)*tile_dim[0]], [tiler*tile_dim[0], tiler*tile_dim[0]], color=[0, 1, 0])
        ax.plot([tilec*tile_dim[0], (tilec+1)*tile_dim[0]], [(tiler+1)*tile_dim[0], (tiler+1)*tile_dim[0]], color=[0, 1, 0])
    ax.set_xlim(-5, image.shape[1]+5)
    ax.set_ylim(image.shape[0]+5, -5)
    ax.axis('off')
    return fig, ax

def show_thresholds(image, t, t_labels=("KI", "Otsu")):
    """Plot image histogram and thresholds
    
    Inputs:
    image: nd array
        Array of pixel values.
    t: list
        List of threshold values.
    tile_labels: tuple
        Tuple of threshold labels for legend.
    Outputs:
    fig: handle to figure
    ax: handle to axis
    """
    accuracy = 200
    h, bin_edges = np.histogram(image[~np.isnan(image)], bins=accuracy, density=True)
    bin_width = bin_edges[1]-bin_edges[0]
    g = np.arange(bin_edges[0]+bin_width/2.0, bin_edges[-1], bin_width)
    fig, ax = plt.subplots()
    ax.bar(g, h, width=bin_width, color=[0.5, 0.5, 0.5, 0.5], edgecolor=[0.5, 0.5, 0.5])
    ps = []
    for t_i, t_value in enumerate(t):
        if "KI" in t_labels[t_i]:
            color = [1, 0.5, 0]
        elif "Otsu" in t_labels[t_i]:
            color = [0, 0.7, 0]
        if "tiled" in t_labels[t_i]:
            linestyle = "dashed"
        else:
            linestyle = "solid"
        p = ax.plot([t_value, t_value], [0, np.nanmax(h)], color=color, linestyle=linestyle)
        ps.append(p[0])
    _ = fig.legend(tuple(ps), t_labels, 'upper right', framealpha=1)
    return fig, ax

def calc_tdict(band_names, sar_image=None, segments=None, thresh_file=None, directory_figure=None, source=['pixels', 'segments'], approach=['tiled', 'global'], t_method=['KI', 'Otsu'], tile_dim=[200, 200], n_final=5, hand_matrix=None, hand_t=100):
    """ Caluclate thresholds for sar_image bands and/or segments, and save to dict
    
    Inputs:
    band_names: list
        List of band names. 
        If sar_image is not None, list should correspond with bands of sar_image.
        If segments is not None, band_names should be in segments.columns.
    sar_image: ndarry or None (default=None)
        Array of SAR pixel values.
    segments: geopandas GeoDataFrame or None (default=None)
        GeoDataFrame of segmetns with band properties.
    thresh_file: str or None (default=None)
        If not None, threshold dictionary is saved to specified path.
    directory_figure: str or None (default=None)
        If not None, fiture is saved to specified path.
    source: list (default=['pixels', 'segments'])
        List specifying whether to consider pixels and/or segments for threshold selection.
    approach: list (default=['tiled', 'global'])
        List specifying whether to apply tiled and/or global thresholding.
    selection: str (default='Martinis')
        Method for tile selection. Currently only option is 'Martinis'.
    t_method: list (default=['KI', 'Otsu'])
        List of thresholds to calculate. Should contain one or both of "KI", "Otsu".
    tile_dim: list (default=[200, 200])
        Dimension of tiles.
    n_final: int (default=5)
        Number of tiles to select.
    hand_matrix: ndarray or None (default=None)
        Array of HAND values.
    hand_t: float (default=100)
        Maximum HAND value allowed for threshold selection.
    Outputs:
    t_dict: dict
        Dictionary of threshold values.
    """
    # Calculate threshold values
    t_dict = {}
    for source_value in source:
        t_dict[source_value] = {}
        if source_value == 'pixels':
            for approach_value in approach:
                t_dict[source_value][approach_value] = {}
                if "KI" in t_method:
                    t_dict[source_value][approach_value]['KI'] = {}
                if "Otsu" in t_method:
                    t_dict[source_value][approach_value]['Otsu'] = {}
                if approach_value == 'tiled':
                    for ib, band in enumerate(band_names):
                            t_values, _ = tiled_thresholding(sar_image[ib], selection='Martinis', t_method=['KI', 'Otsu'], tile_dim=tile_dim, n_final=n_final, hand_matrix=hand_matrix, hand_t=hand_t, directory_figure=directory_figure, incomplete_tile_warning=True)
                            if "KI" in t_method:
                                t_dict[source_value][approach_value]['KI'][band] = t_values[0]
                            if "Otsu" in t_method:
                                t_dict[source_value][approach_value]['Otsu'][band]  = t_values[1]
                elif approach_value == 'global':
                    if "KI" in t_method:
                        for ib, band in enumerate(band_names):
                            t_dict[source_value][approach_value]["KI"][band] = apply_ki(sar_image[ib], accuracy=200)
                    if "Otsu" in t_method:
                        for ib, band in enumerate(band_names):
                            t_dict[source_value][approach_value]["Otsu"][band] = apply_otsu(sar_image[ib], accuracy=200)
        else: # if source_value == 'segments'
            approach_value = 'global'
            t_dict[source_value][approach_value] = {}
            if "KI" in t_method:
                t_dict[source_value][approach_value]["KI"] = {}
                for band in band_names:
                        t_dict[source_value][approach_value]["KI"][band] = apply_ki(segments[band], accuracy=200)
            if "Otsu" in t_method:
                t_dict[source_value][approach_value]["Otsu"] = {}
                for band in band_names:
                        t_dict[source_value][approach_value]["Otsu"][band] = apply_otsu(segments[band], accuracy=200)
    # Save to file
    if thresh_file:
        with open(thresh_file, "wb") as handle:
            pickle.dump(t_dict, handle)
    # Make plot of all threshold values
    if directory_figure:
        for band in band_names:
            leghandles = []
            leglabels = []
            fig, ax = plt.subplots()
            xmin = []
            xmax = []
            if segments:
                h, bins = np.histogram(segments[band], bins=200, density=True)
                g = np.arange(bins[0]+(bins[1]-bins[0])/2, bins[-1], (bins[1]-bins[0]))
                leghandles.append(ax.plot(g, h, color='blue')[0])
                leglabels.append("h segments")
                xmin.append(np.min(segments[band]))
                xmax.append(np.max(segments[band]))
            if sar_image is not None:
                h, bins = np.histogram(sar_image[band_names.index(band)].ravel(), bins=200, density=True)
                g = np.arange(bins[0]+(bins[1]-bins[0])/2, bins[-1], (bins[1]-bins[0]))
                leghandles.append(ax.plot(g, h, color='lightblue')[0])
                leglabels.append("h pixels")
                xmin.append(np.min(sar_image[band_names.index(band)]))
                xmax.append(np.max(sar_image[band_names.index(band)]))
            ax.set_xlim(np.min(np.array(xmin)), np.max(np.array(xmax)))
            M = 0.95 * ax.get_ylim()[1]
            for source_value in source:
                if source_value == 'pixels':
                    for approach_value in approach:
                        for t_method_value in t_method:
                            if t_method_value == 'KI':
                                color = 'red'
                            else:
                                color = 'green'
                            if approach_value == 'global':
                                linestyle = 'dashed'
                            else: 
                                linestyle = 'dotted'
                            leghandles.append(ax.plot([t_dict[source_value][approach_value][t_method_value][band], \
                                       t_dict[source_value][approach_value][t_method_value][band]], [0, M], color=color, linestyle=linestyle)[0])
                            leglabels.append(source_value + ' ' + approach_value + ' ' + t_method_value)
                else:
                    approach_value = 'global'
                    for t_method_value in t_method:
                        if t_method_value == 'KI':
                            color = 'red'
                        else:
                            color = 'green'
                        leghandles.append(ax.plot([t_dict[source_value][approach_value][t_method_value][band], \
                                                   t_dict[source_value][approach_value][t_method_value][band]], [0, M], color=color)[0])
                        leglabels.append(source_value + ' ' + approach_value + ' ' + t_method_value)
            fig.legend(leghandles, leglabels, framealpha=1)
            plt.savefig(os.path.join(directory_figure, "Thresholding_HistWithTs_{}.png".format(band)))
    # Return
    return t_dict