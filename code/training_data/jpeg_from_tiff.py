#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sanja7s


"""

import os
import rasterio
import imageio
import numpy as np

def raster_to_rgb(in_dir, out_dir, img_name):

	file_name = in_dir + img_name

	with rasterio.open(file_name) as src:
		imt = src.read(1) # first band
		im = np.zeros((imt.shape[0], imt.shape[1], 3), "uint8")

		for band_id in range(src.count - 1):	# skip alpha channel
			band = src.read(band_id+1)
			im[:,:,band_id] = band

	out_name = img_name.replace(".tif", ".jpg")
	imageio.imwrite(out_dir + out_name, im)



def batch_transform(in_dir, out_dir):
	""" JPEGs of all tiffs in in_dir into a separate folder in out_dir """

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)   
	
	# recursively iterate through all the files f in inDir
	for root, subdirs, fl in os.walk(in_dir):
		for f in fl:
			if f.endswith(".tiff") or f.endswith(".tif") or f.endswith(".png"):
				print('Creating RGB image ', f)
				raster_to_rgb(in_dir, out_dir, f)


# only for one image
# sat_img = "S2B_MSIL2A_20181024T102059_N0209_R065_T32TNR_20181024T171522_RGB_cropped/"
# in_dir = "preprocessed/training_images/2A_imagelets/" + sat_img
# out_dir = "preprocessed/training_images/2A_imagelet_jpgs/" + sat_img

# for all images
root_in_dir = "preprocessed/training_images/2A_imagelets/"

for in_dir, subdirs, fl in os.walk(root_in_dir):
	if in_dir == root_in_dir:
		continue
	in_dir= in_dir + "/"
	out_dir = in_dir.replace("2A_imagelets", "2A_imagelet_jpgs")
	batch_transform(in_dir, out_dir)