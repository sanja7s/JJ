#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 15:31:41 2018
@author: sanja7s

split input raster files to a set of imagelets each.
input: in_dir
output: out_dir (in which, a separate dir is 
created for each input image)

"""
import os
import sys
import argparse
import numpy as np
from functools import partial
import rasterio
from rasterio.windows import Window
from rasterio import windows
from pyproj import Proj, transform
from osgeo import osr, gdal


min_tile_size = 50 # pixels, square imagelets
tile_size = 64 # desired imagelet size in pixels



def get_extent(dataset):

	cols = dataset.RasterXSize
	rows = dataset.RasterYSize
	transform = dataset.GetGeoTransform()
	minx = transform[0]
	maxx = transform[0] + cols * transform[1] + rows * transform[2]

	miny = transform[3] + cols * transform[4] + rows * transform[5]
	maxy = transform[3]

	return {
			"minX": str(minx), "maxX": str(maxx),
			"minY": str(miny), "maxY": str(maxy),
			"cols": str(cols), "rows": str(rows)
			}



def create_tiles(minx, miny, maxx, maxy, nx, ny):
	width = maxx - minx
	height = maxy - miny

	matrix = []

	for j in range(ny, 0, -1):
		for i in range(0, nx):

			ulx = minx + (width/nx) * i # 10/5 * 1
			uly = miny + (height/ny) * j # 10/5 * 1

			lrx = minx + (width/nx) * (i + 1)
			lry = miny + (height/ny) * (j - 1)
			matrix.append([[ulx, uly], [lrx, lry]])

	return matrix

def create_tiles7s(minx, miny, maxx, maxy, nx, ny, tile_size):
	width = maxx - minx
	height = maxy - miny

	matrix = []

	for j in range(ny, 0, -1):
		for i in range(0, nx):

			ulx = minx + tile_size * 10 * i # 10/5 * 1
			uly = miny + tile_size * 10 * j # 10/5 * 1

			lrx = minx + tile_size * 10 * (i + 1)
			lry = miny + tile_size * 10 * (j - 1)
			matrix.append([[ulx, uly], [lrx, lry]])

	return matrix

def create_predefined_size_tiles(minx, miny, maxx, maxy, tile_size, cols, rows):
	width = maxx - minx
	height = maxy - miny

	matrix = []

	coef_x = width/cols
	coef_y = height/rows

	i = 0
	j = cols

	while (i + 1) * tile_size* coef_x < rows:
		while (j - 1) * tile_size* coef_y > 0:

			ulx = minx + i * tile_size * coef_x
			uly = miny + j * tile_size * coef_y

			lrx = minx + (i + 1) * tile_size * coef_x
			lry = miny + (j - 1) * tile_size * coef_y
			matrix.append([[ulx, uly], [lrx, lry]])

			i+=1
			j-=1

	return matrix

	
def split(OutDir, file_name, tile_size=tile_size): 
	""" split the given raster file_name into pieces 
		of given size approx. 'tile_size'x'tile_size' pixels each
		if a piece smaller than 'min_tile_size' pixels on any dim, don't save
	"""
	
	cnt_too_small = 0
	
	raw_file_name = os.path.splitext(os.path.basename(file_name))[0]
	
	output_path = os.path.join(OutDir, raw_file_name)
	if not os.path.exists(output_path):
		os.makedirs(output_path)   
	
	driver = gdal.GetDriverByName('GTiff')
	dataset = gdal.Open(file_name)

	srcbands = []

	for band in range( dataset.RasterCount ):
		band += 1
		#print ("[ GETTING BAND ]: ", band)
		srcband = dataset.GetRasterBand(band)
		if srcband is None:
			continue
		else:
			srcbands.append(srcband)

	
	transform = dataset.GetGeoTransform()
	extent = get_extent(dataset)

	print ("Extent", extent)
	

	cols = int(extent["cols"])
	rows = int(extent["rows"])
	#print("Columns: ", cols)
	#print("Rows: ", rows)

	nx = int(cols / 100)-1
	ny = int(rows / 100)-1
	#print("nx: ", nx)
	#print("ny: ", ny)    

	minx = float(extent["minX"])
	maxx = float(extent["maxX"])
	miny = float(extent["minY"])
	maxy = float(extent["maxY"])

	width = maxx - minx
	height = maxy - miny
	#print("GCD", osr.gcd(round(width, 0), round(height, 0)))
	#print("Width", width)
	#print("Height", height)


	xOrigin = transform[0]
	yOrigin = transform[3]
	pixelWidth = transform[1]
	pixelHeight = -transform[5]
	print(xOrigin, yOrigin, pixelWidth, pixelHeight)


	nx = int(cols / tile_size)
	ny = int(rows / tile_size)
	tiles = create_tiles7s(minx, miny, maxx, maxy, nx, ny,tile_size)
	print ((minx, miny, maxx, maxy, tile_size))
	print ((width, height, nx, ny, cols, rows))
	# tiles = create_predefined_size_tiles(minx, miny, maxx, maxy, tile_size, cols, rows)

	# print (tiles)

	# tiles= []
	
	def tile_too_small(w, h):
		#print (w,h)
		return (w < min_tile_size or h < min_tile_size)


	tile_num = 0
	for tile in tiles:

		minx = tile[0][0]
		maxx = tile[1][0]
		miny = tile[1][1]
		maxy = tile[0][1]  
		#tile_width = maxx - minx
		#tile_height = maxy - miny

		p1 = (minx, maxy)
		p2 = (maxx, miny)

		i1 = int((p1[0] - xOrigin) / pixelWidth)
		j1 = int((yOrigin - p1[1])  / pixelHeight)
		i2 = int((p2[0] - xOrigin) / pixelWidth)
		j2 = int((yOrigin - p2[1]) / pixelHeight)
		#print(i1, j1)
		#print(i2, j2)

		new_cols = i2-i1
		new_rows = j2-j1
		#print(new_cols,new_rows)
		
		if tile_too_small(new_cols, new_rows):
			cnt_too_small += 1
			continue

		data = []
		for srcband in srcbands:
			data.append(srcband.ReadAsArray(i1, j1, new_cols, new_rows))
		#print(data)

		new_x = xOrigin + i1*pixelWidth
		new_y = yOrigin - j1*pixelHeight
		#print(new_x, new_y)

		new_transform = (new_x, transform[1], transform[2], new_y, transform[4], transform[5])

		output_file_base = raw_file_name + "_" + str(tile_num) + ".tif"
		output_file = os.path.join(OutDir, raw_file_name, output_file_base)
		#print(output_file)

		dst_ds = driver.Create(output_file,
							   new_cols,
							   new_rows,
							   dataset.RasterCount,
							   gdal.GDT_Float32)

		#writting output raster
		for band in range( dataset.RasterCount ):
			band += 1
			dst_ds.GetRasterBand(band).SetNoDataValue(0)
			dst_ds.GetRasterBand(band).WriteArray( data[band-1] )

		tif_metadata = {
			"minX": str(minx), "maxX": str(maxx),
			"minY": str(miny), "maxY": str(maxy)
		}
		dst_ds.SetMetadata(tif_metadata)

		#setting extension of output raster
		# top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
		dst_ds.SetGeoTransform(new_transform)
		wkt = dataset.GetProjection()

		# setting spatial reference of output raster
		srs = osr.SpatialReference()
		srs.ImportFromWkt(wkt)
		dst_ds.SetProjection( srs.ExportToWkt() )

		#Close output raster dataset
		dst_ds = None

		tile_num += 1
		
	dataset = None   
	print('Too small tiles not created ', cnt_too_small)
	

def batchSplit(inDir, outDir):
	print (inDir)
	""" split all tiffs in inDir into a separate folder in ourDir """
	
	# recursively iterate through all the files f in inDir
	for root, subdirs, fl in os.walk(inDir):
		for f in fl:
			if f.endswith(".tiff") or f.endswith(".tif") or f.endswith(".png"):
				#print (f)
				fPath = os.path.join(root, f)
				print('SPLITTING image ', fPath)
				#print('inDir ', inDir)
				#print('outDir ', outDir)
				split(outDir, fPath) 



inDir = "data/satellite_imagery/2A_source_raster"
outDir = "data/satellite_imagery/2A_imagelets/"
batchSplit(inDir, outDir)