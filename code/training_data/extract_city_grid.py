import fiona
from shapely.geometry import mapping, LineString, MultiLineString
import os
import pandas as pd


def city_grid_auto_tiles(file, tile_size, out):
	"""
	input: 
	file: city_shapefile
	tile_size: in meters (we do squares)
	out: wehere we want to grid saved
	---
	output:
	saved shapefile to out location
	"""
	dx = dy = tile_size

	with fiona.open(file, 'r') as ds_in:

		print (ds_in.crs)

		schema = {
		"geometry": "MultiLineString",
		"properties": {"id": "int"}
		 }

		minx, miny, maxx, maxy = ds_in.bounds
		# dx = (maxx - minx) / num_tiles
		num_tilesx = int((maxx - minx) / dx)
		# dy = (maxy - miny) / num_tiles
		num_tilesy = int((maxy - miny) / dy)

		print (minx, miny, maxx, maxy)

		lines = []
		for x in range(num_tilesx + 1):
			lines.append(LineString([(minx + x * dx, miny), (minx + x * dx, maxy)]))
		for y in range(num_tilesy + 1):
			lines.append(LineString([(minx, miny + y * dy), (maxx, miny + y * dy)]))
		grid = MultiLineString(lines)
		
		with fiona.open(out, 'w', driver=ds_in.driver, schema=schema, crs=ds_in.crs) as ds_dst:
			ds_dst.write({'geometry': mapping(grid), "properties": {"id": 0}})

def city_grid(file, num_tiles, out):
	"""
	input: 
	file: city_shapefile
	num_tiles: how many tiles per row and colun we want
	out: wehere we want to grid saved
	---
	output:
	saved shapefile to out location
	"""
	with fiona.open(file, 'r') as ds_in:
		num_tiles = num_tiles
		schema = {
		"geometry": "MultiLineString",
		"properties": {"id": "int"}
		 }

		minx, miny, maxx, maxy = ds_in.bounds
		dx = (maxx - minx) / num_tiles
		dy = (maxy - miny) / num_tiles

		print (minx, miny, maxx, maxy)
		print (dx, dy)

		lines = []
		for x in range(num_tiles + 1):
			lines.append(LineString([(minx + x * dx, miny), (minx + x * dx, maxy)]))
		for y in range(num_tiles + 1):
			lines.append(LineString([(minx, miny + y * dy), (maxx, miny + y * dy)]))
		grid = MultiLineString(lines)
		
		with fiona.open(out, 'w', driver=ds_in.driver, schema=schema, crs=ds_in.crs) as ds_dst:
			ds_dst.write({'geometry': mapping(grid), "properties": {"id": 0}})


def city_grid_centroid_points(city, file, tile_size, out):
	"""
	input: 
	city: name, for image tiles ids
	file: city_shapefile
	tile_size: in meters (we do squares)
	out: where we want to grid saved
	---
	output:
	saved shapefile to out location
	"""
	dx = dy = tile_size

	res = {"image_id":[], "cx":[], "cy":[]}

	with fiona.open(file, 'r') as ds_in:

		print (ds_in.crs)

		schema = {
		"geometry": "MultiLineString",
		"properties": {"id": "int"}
		 }

		minx, miny, maxx, maxy = ds_in.bounds
		# dx = (maxx - minx) / num_tiles
		num_tilesx = int((maxx - minx) / dx)
		# dy = (maxy - miny) / num_tiles
		num_tilesy = int((maxy - miny) / dy)

		print (minx, miny, maxx, maxy)

		i = 0

		lines = []
		for x in range(num_tilesx ):
			for y in range(num_tilesy ):

				# lines.append(LineString([(minx + x * dx, miny), (minx + x * dx, maxy)]))
				# lines.append(LineString([(minx, miny + y * dy), (maxx, miny + y * dy)]))

				res["cx"].append(minx + (x * dx) + dx / 2)
				res["cy"].append(miny + (y * dy) + dy / 2)
				res["image_id"].append(city + str(i))

				i+=1

		pd_res = pd.DataFrame(res)
		pd_res.to_csv(out)

		



# # this was the code if we want to predefine the number of tiles insteas of their size
# num_tiles = 25
# out = "preprocessed/training_images/city_grids/polygon_grid_test.shp"
# file = "data/boundaries/districts/milano/milano_32632.shp"
# city_grid(file, num_tiles, out)

def grid_all_cities():
	tile_size = 150

	cities = ['milano', 'bologna', 'roma', 'firenze', 'torino', 'palermo']

	city_shapes_dir = "data/boundaries/districts/"
	city_grids_dir = "preprocessed/training_images/city_grids/"
	for city in cities:
		city_shape_dir = os.path.join(city_shapes_dir, city)
		for file in os.listdir(city_shape_dir):
			# out = "/polygon_grid_test_auto.shp"
			# file = "data/boundaries/districts/milano/milano_32632.shp"
			if file == city + ".shp":
				out = os.path.join(city_grids_dir, file)
				file = os.path.join(city_shape_dir, file)
				city_grid_auto_tiles(file, tile_size, out)
				print (file)
				print (out)


def points_for_all_cities():
	tile_size = 150

	cities = ['milano', 'bologna', 'roma', 'firenze', 'torino', 'palermo']

	city_shapes_dir = "data/boundaries/districts/"
	city_grids_dir = "preprocessed/training_images/city_grids/"
	for city in cities:
		city_shape_dir = os.path.join(city_shapes_dir, city)
		for file in os.listdir(city_shape_dir):
			# out = "/polygon_grid_test_auto.shp"
			# file = "data/boundaries/districts/milano/milano_32632.shp"
			if file == city + ".shp":
				out = os.path.join(city_grids_dir, city + ".csv")
				file = os.path.join(city_shape_dir, file)
				city_grid_centroid_points(city, file, tile_size, out)
				print (file)
				print (out)

