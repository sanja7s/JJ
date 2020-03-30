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
  * parce the imagery into .geotiff with EPSG:32632 projection
  * place .geotiff under data/training_imagery/satellite

2. Download shapefiles with adminstrative borders of the selected cities
  * place .shp under data/boundaries/districts/

3. Download JJ labels from the repository https://github.com/denadai2/jacobs_urban_planning
  * place under data/labels/

### Data parsing
1. Produce training images (imagelets) by splitting original satellite data into 64x64 images
2. Produce corresponding labels from shapefiles

### Prediction
1. Predict image class for each of the JJ labels




## Licence