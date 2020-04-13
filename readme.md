# Readme file for JJ SAT project

## Short Description
This is a project code for the paper: Predicting Urban Vitality from Satelite imagery.

This repository contains:

1. [The specification](spec.md) for how a standard README should look.
2. A link to a linter you can use to keep your README maintained ([work in progress](https://github.com/RichardLitt/standard-readme/issues/5)).
3. A link to [a generator](https://github.com/RichardLitt/generator-standard-readme) you can use to create standard READMEs.
4. [A badge](#badge) to point to this spec.
5. [Examples of standard READMEs](example-readmes/) - such as this file you are reading.

Standard Readme is designed for open source libraries. Although it’s [historically](#background) made for Node and npm projects, it also applies to libraries in other languages and package managers.


## Table of Contents

- [Install](#install)
- [Requirements](#background)
- [Structure](#structure)
- [Usage](#usage)
	- [Generator](#generator)
- [Badge](#badge)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)



## Background
To make work easier, we choose to reproject all the data to
WGS 84 / UTM zone 32N EPSG:32632, 
as it is easier to work in meter units. 

EPSG:32632 info: Area of use: Between 6°E and 12°E, northern hemisphere between equator and 84°N, onshore and offshore. Algeria. Austria. Cameroon. Denmark. Equatorial Guinea. France. Gabon. Germany. Italy. Libya. Liechtenstein. Monaco. Netherlands. Niger. Nigeria. Norway. Sao Tome and Principe. Svalbard. Sweden. Switzerland. Tunisia. Vatican City State.

Coordinate system: Cartesian 2D CS. Axes: easting, northing (E,N). Orientations: east, north. UoM: m. http://pacificprojections.spc.int/32632


## Install
Requirements for this project include:
```sh
pandas
geopandas
scikit-learn 
xgboost
```

## Structure
1. code
  * training_data
  * regression_prediction
  * regression_data
  * general
  * district_prediction
  * district_data
  * classification_prediction
  * prediction
  * test
2. data
  * boundaries
   * blocks
   * districts
    * italy7s
    * one for each city: bologna, milano, firenze, palermo, roma, torino
  * training_data
   * labels from JJ the project
   * satellite
  * JJ_urban_variables.xlsx
3. preprocessed
  * training_images
  * labels
  * features
  * high_low_examples
  * fig
4. results
  * XGBoost

## Usage

TODO

```sh
$ standard-readme-spec
# Prints out the standard-readme spec
```

### Data collection
1. Download satellite imagery around selected cities (Sentinel)
  * preprocess the imagery into .geotiff with EPSG:32632 projection
  * place .geotiff under `data/training_imagery/satellite/` 

2. Download shapefiles with adminstrative borders of the selected cities
  * place .shp files under `data/boundaries/districts/`. NOTE: We have found that it is best to download the whole regions from Italy in which the respective cities are and then extract each city's shape file. If you will do the same, the regional shapefiles are under `data/boundaries/districts/italy7s/` while the code to extract each city's shapefile is under `code/training_data/extract_city_shapes.py`. Upon running this code for each city, we have the respective shapefiles saved (in the proper projection) under `data/boundaries/districts/city`.

3. Download JJ labels from the repository https://github.com/denadai2/jacobs_urban_planning
  * place under data/labels/

## Data preprocessing
### Training Imagery
1. Crop each satellite image to the area encompassing the city shapefile. This is done using `code/satellite_imagery/crop_raster_using_vector.py` and will save the cropped .geotiffs under `data/satellite_imagery/crop/2A_source_raster/`. 
2. Produce training data (imagelets) by splitting the cropped satellite images into 64x64 px pieces. This is done in several steps: first, using `code/satellite_imagery/split_raster.py`, we create imagelets as .geotiffs, i.e., preserving their spatial information, under `data/satellite_imagery/2A_imagelets/`. Second, we produce .rgb imagelets, i.e., those that are suitable for deep learning extractors, using `code/satellite_imagery/jpeg_from_tiff.py` -- this saves our results under `data/satellite_imagery/2A_imagelets_jpgs/`.
3. Produced .rgb imagery is parsed using VGG-16 extractor (TODO we add code?) and resulting feature vectors are saved under `preprocessed/features/`. Since the feature sizes from the extractor are from 2048 to 4096, we need to reduce them. The code using PCA to reduce this set of features to a predefied number of components (we worked with 16,32) is under `code/training_data/PCA_features_all.py`

### Labels Imagery
1. We need labels for each imagelet. There are several ways to approach the labelling. On the finest granularity level, each imagelet will get the label (for each of the JJ variables) based on the district in which the imagelet is located. On the lower granularity, we can merge the districts into classes (say quartiles or tertiles for each variable). In this case, we first disolve the shapefile with districts along such a defined class and then look for each imagelet, in which area (distrct group) it is found. The difference in resulting labels between these two approaches can come from how we define the imagelet being inside a district/group of districts. Due to the small distrcit sizes and low resolution of the Sentinel imagery we work with, the imagelets will rarely fall completely within a single district. Hence, we here again took two possible approaches. In the first: imagelet is defined to be within a district if it overlaps with more than the 50% (THRESHOLD=.5) of its area with the districts area. In the second approach, we simply assign the imagelet to the district with which it has the highest overlap. This allows to label more imagelets, as the strict criteria in the first case omits many imegelets that overlapped with several districts and with none of them with more than 50% of its size. 

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
