#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Lisa Landuyt
@purpose: Ancillary functions for quickshift segmentation
"""

import os
import numpy as np
from sklearn import preprocessing
import datetime
from skimage import measure
from skimage import segmentation as seg
import rasterio
import ImageFunctions as ipf
    

def apply_quickseg(image, image_bandnames, image_metadata, ratio=1.0, maxdist=4, kernel_window_size=7, directory_output=None):
    """
    Apply quickshift segmentation for the specified set of parameters.
    
    Inputs:
    image: nd array
        Input array for segmentation. Dimensions should be rows x columns x bands.
    image_bandnames: list
        List specifying the band order in image. Possible elements are:
        rVH, rVV, fVH, fVV, rR, fR, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
    image_metadata: dict
        Dictionary specifying image metdata, output of rasterio meta property.
    ratio: float (default=1.0)
        Ratio balancing color-space proximity and image-space proximity, should be between 0 and 1. Higher values give more weight to color-space.
    maxdist: float (default=4)
        Cut-off point for data distances. Higher means fewer clusters.
    kernel_window_size: int (default=7)
        Size of Gaussian kernel used in smoothing the sample density. Higher means fewer clusters. Minimum equals 7x7.
    directory_output: str or None (default=None)
        If not None, output will be saved to specified path.
    Outputs:
    segments_quick: nd array
        Array of segments IDs. 
    """
    # Check image dimensions
    no_rows, no_cols, no_bands = image.shape
    if no_bands > no_rows:
        print("Warning! Image dimensions should be row x column x bands. Current dimensions are {}x{}x{}, which seems wrong. Swapping axes...".format(no_rows, no_cols, no_bands))
        image = np.transpose(image, (1, 2, 0))
        no_rows, no_cols, no_bands = image.shape
    else:
        print("Image dimensions ({}x{}x{}) are valid.".format(no_rows, no_cols, no_bands))
    # Normalize data
    for band_index in np.arange(no_bands):
        if np.nanstd(image[:,:,band_index]) != 1 or np.nanmean(image[:,:,band_index]) != 0:
            band = image[:,:,band_index]
            band[np.isfinite(band)] = preprocessing.StandardScaler().fit_transform(band[np.isfinite(band)].reshape(-1, 1))[:,0]
            image[:,:,band_index] = band
    # Segmentation
    kernel_size = (kernel_window_size - 1)/6
    image_segmented = seg.quickshift(image.astype('double'), ratio=ratio, max_dist=maxdist, kernel_size=kernel_size, convert2lab=False)
    image_segmented += 1 # add 1 to avoid background value 0
    image_segmented = measure.label(image_segmented, connectivity=1)
    num_segments = np.unique(image_segmented).size
    mask = ~np.isnan(image[:,:,0])
    segments_quick = ipf.polygonize(image_segmented, mask=mask, transform=image_metadata["transform"])
    print("{} - {} segments detected.".format(datetime.datetime.now(), num_segments)) 
    # Save output    
    if directory_output:
        output_filename_tiff = os.path.join(directory_output, "Segments_r{}_m{}_k{}.tif".format(ratio, maxdist, kernel_window_size))
        print("Saving raster output to {}...".format(output_filename_tiff))
        ipf.ar2tif(image_segmented, output_filename_tiff, image_metadata["crs"], image_metadata["transform"], dtype=rasterio.int32)
        
        output_filename_shp = os.path.join(directory_output, "Segments_r{}_m{}_k{}.shp".format(ratio, maxdist, kernel_window_size))
        print("Saving features output to {}...".format(output_filename_shp))
        segments_quick.to_file(output_filename_shp)
    # Return
    return segments_quick, image_segmented


def selec_tile(image, image_transform, tile_size, tile_index):
    """
    Select tile from image (for tiled segmentation)
    
    Inputs
    Outputs
    """
    # Get image shape and number of tile rows/cols
    s = image.shape
    if len(s) == 3:
        n_rows, n_cols, n_bands = s
    elif len(s) == 2:
        n_rows, n_cols = s
    nrt, nct = np.ceil(np.array([n_rows, n_cols]) / tile_size).astype('int')
    # Get tile transform and tile array
    i = int(np.floor(tile_index / nct))
    j = tile_index % nct
    tile_transform = tuple(image_transform)
    tile_transform[2] += j * tile_size * tile_transform[0]
    tile_transform[5] += i * tile_size * tile_transform[4]
    tile = image[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size, :]
    # Return
    return tile, tile_transform
