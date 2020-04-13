#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sanja7s

Extract geocoordinates of the centroid of a raster by first
Polygonizing the input raster and then getting the points.y

input: in_dir
output: out_dir (in which, a separate dir is created for each input image)

this does a simple outline shape but the code could be edited to 
polygonize actually and include the raster values.
"""

import os
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import cascaded_union
import geopandas as gp
import pandas as pd

def raster_to_shape_centroid(in_dir, img_name):

	file_name = in_dir + img_name

	mask = None
	with rasterio.open(file_name) as src:
		image = src.read(1) # first band
		results = (
		{'properties': {'raster_val': v}, 'geometry': s}
		for i, (s, v) 
		in enumerate(
			shapes(image, mask=mask, transform=src.transform)))

	geoms = list(results)

	gpd_polygonized_raster  = gp.GeoDataFrame.from_features(geoms)
	gpd_polygonized_raster["single_val"] = 1
	gpd_polygonized_raster = gpd_polygonized_raster.dissolve(by="single_val")

	gpd_polygonized_raster.crs = src.crs
	shape_name = img_name.replace(".tif", "")
	
	geom = gpd_polygonized_raster
	# Centroid property returns a string with x and y separated by a space
	xcoord = geom.centroid.x.values[0]
	ycoord = geom.centroid.y.values[0]

	# print (geom.centroid)
	# print (xcoord, ycoord)

	return {"imageName": shape_name, "centroid":(xcoord, ycoord)}



def batch_extract(in_dir, out_dir):
	""" extract centroid geocoordinate pair for each imagelet """
 
	all_coord = []
	
	# recursively iterate through all the files f in inDir
	for root, subdirs, fl in os.walk(in_dir):
		for f in fl:

			if f.endswith(".tiff") or f.endswith(".tif") or f.endswith(".png"):
				#print (f)
				# fPath = os.path.join(root, f)
				# print('Extracting image ', f)

				all_coord.append(raster_to_shape_centroid(in_dir, f))

				# if i == 100:
				# 	print (all_coord)
				# 	res = pd.DataFrame(all_coord)
				# 	print (res)
				# 	res.to_csv(out_dir+"all_imagelets_geocoord.csv")

	# print (all_coord)
	return all_coord



# code for one selected image only
#sat_img = "S2B_MSIL2A_20181024T102059_N0209_R065_T32TNR_20181024T171522_RGB_cropped/"
#in_dir = "preprocessed/training/2A_imagelets/" + sat_img
#out_dir = "preprocessed/training/2A_imagelet_shapes/" + sat_img
root_in_dir = "preprocessed/training_images/2A_imagelets/"
out_dir = "preprocessed/features/features_all/"
all_res = []

for in_dir, subdirs, fl in os.walk(root_in_dir):
	if in_dir == root_in_dir:
		continue
	in_dir = in_dir + "/"
	print (in_dir)
	all_res += batch_extract(in_dir, out_dir)

all_res = pd.DataFrame(all_res)
print (all_res)
all_res.to_csv(out_dir+"all_imagelets_geocoord.csv")


