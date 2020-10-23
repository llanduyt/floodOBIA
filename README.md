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
