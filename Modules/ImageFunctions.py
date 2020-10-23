#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Lisa Landuyt
@purpose: Ancillary functions for image I/O and manipulation
"""

import os
import rasterio
import numpy as np
from scipy import interpolate
import rasterstats as rs
import geopandas as gpd
import datetime

def tif2ar(infile, window=None, band=None, return_bandnames=False):
    """ Read in a .tif file to a numpy array
    
    Inputs:
    infile: string
        Path + name of input .tif file
    window: pair of tuples
        ((rmin, rmax), (cmin, cmax)) defining the indices of the columns/rows to read
    band: int
        Index (starting from 1) of the band to read. If None, all bands are read
    return_bandnames: bool
        If true, a list containing the band descriptions (if set) will be returned
    Outputs:
    array: nd array
        Image pixel values
    metadata: dict
        Dictionary containing metadata: driver, dtype, nodata, width, height, count, crs, transform
    band_names (if return_bandnames is True): list
        List of band descriptions
    """ 
    if not os.path.isfile(infile):
        print("Error! File {} does not exist.".format(infile))
        if return_bandnames:
            return None, None, None
        return None, None
    with rasterio.open(infile) as src:
        metadata = src.meta
        if src.count == 1:
            band = 1
        array = src.read(band, window=window)
        if return_bandnames:
            band_names = list(src.descriptions)
    if return_bandnames:
        return array, metadata, band_names
    return array, metadata


def ar2tif(array, outfile, crs, transform, dtype=rasterio.float32, band_index=0, band_names=None):
    """ Export a numpy array to an GeoTiff
    
    Inputs:
    array: nd array
       Image pixel values. Should have shape (rows, cols, bands). If not, specify the band_index.
    outfile : string
       Path + name of the output .tif file.
    crs: rasterio.crs.CRS object or int
        CRS object or EPSG code of crs.
    transform: Affine
        Transformation from pixel to geographic coordinates.
    dtype: rasterio dtype or None (default=rasterio.float32)
       If None, equal to the raster dtype.
    band_index: int (default=0)
        Index of bands in the shape tuple. Possible values: 0, 2.
    band_names: list or None (default=None)
        List of descriptions for the bands.
    """
   
    s = array.shape
    if len(s) == 2:
        rows, cols = s
        bands = 1
    elif len(s) > 2:
        if band_index == 2:
            rows, cols, bands = s 
        elif band_index == 0:
            bands, rows, cols = s
        else:
            print("Error! Band_index value invalid. Should be 0 or 2, is {}. Aborting.".format(band_index))
    elif len(s) > 3:
        print("Error! Array shape invalid. Aborting.")
    if band_names and len(band_names) != bands:
        print("Error! List band_names should have length equal to no. of bands. Currently {} vs. {}. No band names saved.".format(len(band_names), bands))
        band_names = None

    if str(np.dtype(array.dtype)) != dtype:
        dtype = str(np.dtype(array.dtype))
    
    with rasterio.open(outfile, 'w', driver='GTiff', height=rows, width=cols, count=bands, crs=crs, dtype=dtype, transform=transform) as dst:
            if bands == 1:
                dst.write(array, 1)
                if band_names is not None:
                    dst.set_band_description(1, band_names[0])
            else:
                for b in range(bands):
                    if band_index == 2:
                        dst.write_band(b+1, array[:,:,b])
                    elif band_index == 0:
                        dst.write_band(b+1, array[b,:,:])
                    if band_names is not None:
                        dst.set_band_description(b+1, band_names[b])

 
def rescale01(array, lowerP=0, upperP=100, band_index_in=0, band_index_out=0, include_stats=False):
    """ 
    rescale the values of an array to [0, 1]. 
    
    Inputs:
    array: nd array
        Values to rescale.
    lowerP: float (default=0)
        Lower percentile to clip values to.
    upperP: float (default=100)
        Upper percentile to clip values to.
    band_index_in: int (default=0)
        Index of band dimension in input array.
    band_index_out: int (default=0)
        Desired index of band dimension for output array.
    include_stats: bool (default=False)
        Whether to return the percentile valeus for each band.
    Outputs:
    array_rescaled: ndarray
        Rescaled values
    stats (if include_stats is True): dict
        Lower and upper percentile values for each band.
    """
    if len(array.shape) == 3:
        if band_index_in == 0:
            b, r, c = array.shape
        elif band_index_in == 2:
            r, c, b = array.shape
            array = np.transpose(array, axes=(2,1,0))
    else:
        b = 1
        r, c = array.shape
        array = array.reshape(b,r,c)
    array_rescaled = np.copy(array)
    pvalues = []
    Pvalues = []
    for i, band in enumerate(array_rescaled):
        # Clip
        p = np.percentile(band, lowerP)
        P = np.percentile(band, upperP)
        pvalues.append(p)
        Pvalues.append(P)
        band[band > P] = P
        band[band < p] = p
        # Rescale
        array_rescaled[i] = (band - np.min(band)) / np.ptp(band)
        del band
    if b == 1:
        array_rescaled = array_rescaled.reshape(r,c)
    elif band_index_out == 2:
        array_rescaled = np.transpose(array_rescaled, axes=(1,2,0))
    if include_stats:
        stats = {'lower perc': pvalues, 'upper perc': Pvalues}
        return array_rescaled, stats
    else:
        return array_rescaled
    
    
def polygonize(image, mask=None, transform=None):
    """Polygonize image to geopandas GeoDataFrame with DN attribute
    
    Inputs:
    image: nd array
        Image values to retrieve objects from.
    mask: nd array or rasterio Band or None (default=None)
        Mask used to exclude pixels (value False or 0) from object generation.
    transform: Affine or None (default=None)
        Transformation from pixel to geographic coordinates.
    Outputs:
    segments: geopandas GeoDataFrame
        GeoDataFrame containing generated objects with DN attribute.
    """
    shape_pairs = rasterio.features.shapes(image.astype(np.int32), mask=mask, transform=transform, connectivity=4)
    segments_list = [{'properties': {'DN': dn}, 'geometry': g} for i, (g, dn) in enumerate(shape_pairs)]
    return gpd.GeoDataFrame.from_features(segments_list).astype({'DN': 'int32'})


def rasterize_toarray(segments, burn_value, image_shape, image_transform=None, image_dtype=None, nodata_value=0):
    """Rasterize segments to numpy array
    
    Inputs:
    segments: geopandas GeoDataFrame
        GeoDataFrame containing objects to retrieve raster from.
    burn_value: str, int or float
        Column label or fixed value used to burn.
    image_shape: tuple of 2 int
        Shape of output numpy array
    image_transform: Affine or None (default=None)
        Transformation from pixel to geographic coordinates.
    image_dtype: rasterio/numpy dtype or None (default=None)
        Data type of output numpy array.
    nodata_value: int, float or None (default=0)
        Value used to fill array where not covered by input objects.
    Outputs:
    image: nd array
        Rasterized objects.
    """
    if type(burn_value) is str:
        shape_pairs = ((g, value) for g, value in zip(segments.geometry, segments[burn_value]))
    else:
        shape_pairs = ((g, burn_value) for g in segments.geometry)
    image = rasterio.features.rasterize(shapes=shape_pairs, out_shape=image_shape, fill=nodata_value, transform=image_transform, dtype=image_dtype) 
    return image


def rasterize_tofile(segments, burn_value, image_filename, image_meta, fill=None):
    """Rasterize segments to GeoTiff
    
    Inputs:
    segments: geopandas GeoDataFrame
        GeoDataFrame containing objects to retrieve raster from.
    burn_value: str, int or float
        Column label or fixed value used to burn.
    image_filename: str
        Path for output file.
    image_meta: dict
        Dictionary of rasterio metadata (.meta) to construct output GeoTiff.
    fill: int, float or None (default=None)
        Value used to fill array where not covered by input objects. If None, equals image nodata value.
    Outputs:
    success: bool
        True if rasterization successful.
    """
    if type(burn_value) is str:
        shape_pairs = ((g, value) for g, value in zip(segments.geometry, segments[burn_value]))
    else:
        shape_pairs = ((g, burn_value) for g in segments.geometry)
    if not fill:
        if not image_meta["nodata"]:
            if image_meta["dtype"] == rasterio.uint8:
                image_meta["nodata"] = 255
            elif image_meta["dtype"] == rasterio.uint16:
                image_meta["nodata"] = 65535
            elif image_meta["dtype"] == rasterio.uint32:
                image_meta["nodata"] = 4294967295
            elif image_meta["dtype"] == rasterio.int16:
                image_meta["nodata"] = -32768
            else:
                image_meta["nodata"] = -9999
        fill = image_meta["nodata"]
    out_image = rasterio.features.rasterize(shapes=shape_pairs, out_shape=(image_meta["height"], image_meta["width"]), fill=fill, transform=image_meta["transform"], dtype=image_meta["dtype"]) 
    with rasterio.open(image_filename, "w", **image_meta) as dst:
        dst.write(out_image, indexes=1)
    return True

def extract_features(image, image_label, segments, stats=["mean"], band_num=1, image_transform=None, nodata=-9999):
    """
    Extract features by calculating raster statistics
    
    Inputs:
    image: nd array or str 
        Image to retrieve statistics from, array or path to image file.
    image_label: str
        Label describing content of image.
    segments: geopandas GeoDataFrame or str
        Segments to retrieve statistics for, GeoDataFrame or path to segments file.
    stats: list (default=["mean"])
        List of statistics to calculate.
    band_num: int (default=1)
        Image band number to use (counting from 1).
    image_transform: Affine or None (default=None)
        Transformation from pixel to geographic coordinates, only required if image is ndarray.
    nodata: float or None (default=-9999)
        Value to assign if no image data to retrieve.
    Outputs:
    image_label: str
    statistics: list of dicts
    """
    print("{} - {} - Extracting features for {}".format(datetime.datetime.now(), os.getpid(), image_label))
    return image_label, rs.zonal_stats(segments, image, stats=stats, affine=image_transform, band_num=band_num, nodata=nodata)


def warp_to_roi(input_filename, output_filename, roi_filename):
    """Warp image to ROI
    
    Inputs:
    input_filename: str
    Path to input filename
    output_filename: str
    Path to store output file
    roi_filename: str
    Path to file used to retrieve ROI.
    Outputs:
    success: bool
        True if warping was successful.
    """
    with rasterio.open(roi_filename) as src:
        roi_transform = src.transform
        roi_crs = src.crs
    with rasterio.open(input_filename) as src:
        src_transform = src.transform
        src_crs = src.crs
        src_array = src.read()
    rasterio.warp.reproject(src_array, output_filename, src_transform=src_transform, src_crs=src_crs, dst_transfrom=roi_transform, dst_crs = roi_crs, resampling=rasterio.warp.Resampling.nearest)
    return os.path.exists(output_filename)

