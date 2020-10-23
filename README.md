# floodOBIA
An object-based clustering approach for SAR-based flood mapping.

This repository contains the source codes used in the following publication:

Landuyt et al., (in review). "Flood mapping in vegetated areas using an unsupervised clustering approach on Sentinel-1 and -2 imagery". Remote Sensing.

### Prerequisites
This repository is mainly built using the following packages:
```
python 3.8.2
rasterio 1.1.7
geopandas 0.8.1
pygeos 0.8
rasterstats 0.15
scikit-image 0.17.1
```

### Run the code
The main file is OBIAflood.py. This script requires the path to a .csv file specifying the settings as an argument. 
An example settings file, Settings.csv, is included.

#### Input
A description of all parameters is given in the third column of Settings.csv

The main input files are:
- A stack of 2 SAR images, both in VV and VH polarization. The following band order is assumed: reference_VH, reference_VV, flood_VH, flood_VV.
- A stack of optical bands (cloudfree image or composite). The following band order is assumed: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12.
- DEM of the ROI. In the paper, the hydrologically conditioned SRTM DEM was used.
- LC of the ROI. The code assumes Copernicus GLS land cover codes.

#### Output
The resulting classification is saved both as a .shp and a .tif file. In the .shp, the attribute "model" contains the final classification. In the final classification, the following classes (with label) are present: 
- 0 = dry land (DL)
- 1 = permanent water (PW)
- 2 = open flooding (OF)
- 3 = flooded vegetation (FV)
- 4 = low-lying areas considered as OF
- 5 = forests possibly hiding flood
