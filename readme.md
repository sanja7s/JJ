# Readme file for JJ SAT project

## Short Description
This is a project code for the paper: Predicting Urban Vitality from Satelite imagery.


## Table of Contents

- [Background](#background)
- [Install](#install)
- [Structure](#structure)
- [Usage](#usage)
	- [Generator](#generator)
- [Badge](#badge)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)


## Background
All the geospatial data in this project are reprojected to [WGS 84 / UTM zone 32N EPSG:32632](http://pacificprojections.spc.int/32632), as it is the default for Italy and it works in meter units. 

In this project, we experimented with predicting both vitality for both imagelets and whole distritcts.


## Install
Requirements for this project include:
```sh
pandas
geopandas
scikit-learn 
xgboost
```

## Structure
code
  * [satellite_imagery](code/satellite_imagery)
  * [training_data](code/training_data)
  * [prediction](code/prediction)
  * [general](code/general)
  * [bing](code/bing)

data
  * [boundaries](data/boundaries)
  * [satellite_imagery](data/satellite_imagery)
  * [labels](data/labels)

preprocessed
  * [training_data](preprocessed/training_data)

results
  * XGBoost

## Usage

### Data collection
1. Download satellite imagery around selected cities (Sentinel-2) and preprocess the imagery into .geotiff with EPSG:32632 projection. Then place the .geotiff under `data/training_imagery/satellite/`.

2. Download shapefiles with adminstrative borders of the selected cities and place the .shp files under `data/boundaries/districts/`. NOTE: We have found that it is best to download the whole regions from Italy in which the respective cities are and then extract each city's shape file. If you will do the same, the regional shapefiles are under `data/boundaries/districts/italy7s/` while the code to extract each city's shapefile is under `code/general/extract_city_shapes.py`. Upon running this code for each city, we have the respective shapefiles saved (in the proper projection) under `data/boundaries/districts/city`.

3. Download JJ labels from the repository of the paper [The Death and Life of Great Italian Cities: A Mobile Phone Data Perspective](https://github.com/denadai2/jacobs_urban_planning) and place under `data/labels/`.

## Data preprocessing

### Satellite Imagery (Imagelets)
1. Crop each satellite image to the area encompassing the city shapefile. This is done using `code/satellite_imagery/crop_raster_using_vector.py` and will save the cropped .geotiffs under `data/satellite_imagery/crop/2A_source_raster/`. 
2. Produce training data (imagelets) by splitting the cropped satellite images into 64x64 px pieces. This is done in two steps: first, using `code/satellite_imagery/split_raster.py`, we create imagelets as .geotiffs, i.e., we preserve their spatial information, under `data/satellite_imagery/2A_imagelets/`. Second, we produce .rgb imagelets, i.e., those that are suitable for deep learning extractors, using `code/satellite_imagery/jpeg_from_tiff.py` -- this saves our results under `data/satellite_imagery/2A_imagelets_jpgs/`.

### Deep Learning Features (Imagelets)
3. Produced .rgb imagery is parsed using VGG-16 extractor (TODO `code/training_data/feature_extractor/`) and resulting feature vectors are saved under `preprocessed/features/imagelets`. Since the size of feature from the extractor ranges from 2048 to 4096, we need to reduce them. The code using PCA to reduce this set of features to a predefined number of components (we experimented with number of features ranging from 8 to 64) is under `code/training_data/imagelets/PCA_features.py`

### Deep Learning Features (Districts)
4. To obtain district-level features, the corresponding imagelet features are aggregated using the code under ... 

### Labels
5. We need labels for both classification and regression for each imagelet/district. For districts, this is a more straightforward process: each district takes its regression label from the original label data under `data/labels/`, while its classification label is defined based on the quartile its regression label takes among all the district labels. 


6. For imagelets, the situation is a bit more complex. 

	Each imagelet will get the regression label (for each of the JJ variables) based on the district in which the imagelet is located. There are two ways to define whether the imagelet is located in a district. Due to the small distrcit sizes and the low resolution of the Sentinel imagery we work with, the imagelets will rarely fall completely within a single district. Hence, thek two possible approaches are:
		1. imagelet is located within a district with which it overlaps with more than the 50% (THRESHOLD=.5) of its area with the districts area,
		2. imagelet is assigned to the district with which it has the highest overlap. This allows to label more imagelets, as the strict criteria in the first case omits many imegelets that overlapped with several districts and with none of them with more than 50% of its size. 


	For classificaton labels, we merge the districts into classes (the quartiles for each variable) and then disolve the shapefile along such a defined class. We then look for each imagelet, in which area (distrct group) it is located, again having two possible defitions as above. 


The steps to achieve these 4 possible types of labeling are:
1) produce shapefiles from imagelets using `code/satellite_imagery/polygonize_raster.py`. This saves a new set of files under `data/satellite_imagery/2A_imagelet_shapes/`. We do this since it is easier to calculate the overlap with district sizes in the next step using such shapefiles than rasters of imagelets. 2) For a finer granularity labelling, we use `code/district_data/training_data_assign_labels_districts.py`, which assigns each imagelet('s shape) to the district. The LABELING_METHOD in this script can be set to "threshold" or "maximum" depedning on which of the two above discussed approaches to labelling we take. The "threshold" method is more strict. This script saves files with imagelet names and their district labels under `preprocessed/district_labels/`.


### Prediction
1. Predict image class for each of the JJ labels

### Generator

To use the generator, look at [generator-standard-readme](https://github.com/RichardLitt/generator-standard-readme). There is a global executable to run the generator in that package, aliased as `standard-readme`.



To add in Markdown format, use this code:

```
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)
```



## Maintainers

[@sanja7s](https://github.com/sanja7s)



## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
