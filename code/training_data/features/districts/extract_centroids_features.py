import pandas as pd
import geopandas as gp
import os
from shutil import copy2
from pathlib import Path

"""
collate the shapefiles of the cities and
find centroid of each district. output
a .csv with city_district --> (x, y)
all in EPSG 32632
"""


CRS = '32632'

def batch_extract_centroids(city_name):
	label_lst = []
	print('EXTRACTING  ', city_name)

	try:
		label = \
			gp.read_file(shapes_dir + city_name + "_epsg_" +CRS+ ".shp")
	except:
		return None

	max_label = label["district"].max()
	min_label = label["district"].min()
	print (min_label, max_label)

	label["city_district"] = \
		label["district"].astype(str)\
		.apply(lambda x: city_name + "_" + x)

	label["centroid_point"] = label["geometry"].centroid
	label["centroid_x"] = label["centroid_point"].apply(lambda p: p.x)
	label["centroid_y"] = label["centroid_point"].apply(lambda p: p.y)
	label_df = label[["city_district","centroid_x", "centroid_y"]]

	# print (label_df) 
	return label_df


def extract_one_city(city_name = "milano"):

	try: 
		labels = batch_extract_centroids(city_name)
	except Exception as e:
		print (e)

	return labels

shapes_dir = "data/boundaries/districts/"
out_file = \
	"preprocessed/training_data/features/districts/all_districts_centroids.csv"


# this shapes dir is created by us by dissolving the block shapefiles
cities = ['bologna', 'firenze', 'torino', 'palermo', 'roma', 'milano'] 
res = pd.DataFrame()
for city_name in cities:
	df_one = extract_one_city(city_name)
	res = pd.concat([res, df_one])
res.to_csv(out_file, index=0)







