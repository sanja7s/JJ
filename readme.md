# Readme file for JJ SAT project


## Requirements
```bash
pandas
geopandas
scikit-learn 
xgboost
```

To make work easier, we choose to reproject all the data to
WGS 84 / UTM zone 32N EPSG:32632, 
as it is easier to work in meter units. 

EPSG:32632 info: Area of use: Between 6°E and 12°E, northern hemisphere between equator and 84°N, onshore and offshore. Algeria. Austria. Cameroon. Denmark. Equatorial Guinea. France. Gabon. Germany. Italy. Libya. Liechtenstein. Monaco. Netherlands. Niger. Nigeria. Norway. Sao Tome and Principe. Svalbard. Sweden. Switzerland. Tunisia. Vatican City State.

Coordinate system: Cartesian 2D CS. Axes: easting, northing (E,N). Orientations: east, north. UoM: m. http://pacificprojections.spc.int/32632


## Structure

1. code
  * training_data
  * general
  * prediction
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


## Procedure

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
1. Crop each satellite image to the area encompassing the city shapefile. This is deon using `code/general/crop_raster_using_vector.py` and will save the cropped .geotiffs under `preprocessed/training/2A_source_raster/`. 
2. Produce training data (imagelets) by splitting the cropped satellite imeages into 64x64 px pieces. This is done in several steps: first, using `code/training_data/split_raster.py`, we create imagelets as .geotiffs, i.e., preserving their spatial information, under `preprocessed/training_images/2A_imagelets/`. Second, we produce .rgb imagelets, i.e., those that are suitable for deep learning extractors, using `code/training_data/jpeg_from_tiff.py` -- this saves our results under `preprocessed/training_images/2A_imagelets_jpgs/`.
3. Produced .rgb imagery is parsed using VGG-16 extractor (TODO we add code?) snf resulting feature vectors are saved under `preprocessed/features/`. Since the feature sizes from the extractor are from 2048 to 4096, we need to reduce them. The code using PCA to reduce this set of features to a predefied number of components (we worked with 16,32) is under `code/training_data/PCA_features_all0.py`

### Labels Imagery
1. We need labels for each imagelet. There are several ways to approach the labelling. On the finest granularity level, each imagelet will get the label (for each of the JJ variables) based on the district in which the imagelet is located. On the lower granularity, we can merge the districts into classes (say quartiles or tertiles for each variable). In this case, we first disolve the shapefile with districts along such a defined class and then look for each imagelet, in which area (distrct group) it is found. The difference in resulting labels between these two approaches can come from how we define the imagelet being inside a district/group of districts. Due to the small distrcit sizes and low resolution of the Sentinel imagery we work with, the imagelets will rarely fall completely within a single district. Hence, we here again took two possible approaches. In the first: imagelet is defined to be within a district if it overlaps with more than the 50% (THRESHOLD=.5) of its area with the districts area. In the second approach, we simply assign the imagelet to the district with which it has the highest overlap. This allows to label more imagelets, as the strict criteria in the first case omits many imegelets that overlapped with several districts and with none of them with more than 50% of its size. 

The steps to achieve these 4 possible types of labeling are:
1) produce shapefiles from imagelets using `code/training_data/polygonize_raster.py`. This saves a new set of files under `preprocessed/training_images/2A_imagelet_shapes/`. We do this since it is easier to calculate the overlap with district sizes in the next step using such shapefiles than rasters of imagelets. 2) For a finer granularity labelling, we use `code/district_data/training_data_assign_labels_districts.py`, which assigns each imagelet('s shape) to the district. The LABELING_METHOD in this script can be set to "threshold" or "maximum" depedning on which of the two aobe discussed approaches to labelling we take. The "threshold" method is more strict. This script saves files with imagelet names and their district labels under `preprocessed/district_labels/`.


### Prediction
1. Predict image class for each of the JJ labels




## Licence