#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sanja7s

Polygonize input raster files to a set of shapes each.
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


def raster_to_shape(in_dir, out_dir, img_name):

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
	shape_name = img_name.replace(".tif", ".shp")
	gpd_polygonized_raster.to_file(out_dir + shape_name)



def batch_polygonize(in_dir, out_dir):
	""" polygonize all tiffs in inDir into a separate folder in ourDir """

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)   
	
	# recursively iterate through all the files f in inDir
	for root, subdirs, fl in os.walk(in_dir):
		for f in fl:
			if f.endswith(".tiff") or f.endswith(".tif") or f.endswith(".png"):
				#print (f)
				# fPath = os.path.join(root, f)
				print('Polygonizing image ', f)
				#print('inDir ', inDir)
				#print('outDir ', outDir)
				raster_to_shape(in_dir, out_dir, f)


# code for one selected image only
#sat_img = "S2B_MSIL2A_20181024T102059_N0209_R065_T32TNR_20181024T171522_RGB_cropped/"
#in_dir = "preprocessed/training/2A_imagelets/" + sat_img
#out_dir = "preprocessed/training/2A_imagelet_shapes/" + sat_img

root_in_dir = "preprocessed/training/2A_imagelets/"

for in_dir, subdirs, fl in os.walk(root_in_dir):
	if in_dir == root_in_dir:
		continue
	in_dir = in_dir + "/"
	out_dir = in_dir.replace("2A_imagelets", "2A_imagelet_shapes")
	print (in_dir, out_dir)
	batch_polygonize(in_dir, out_dir)