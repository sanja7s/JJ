import geopandas as gp
import os
import matplotlib.pyplot as plt
import numpy as np

import earthpy as et
import earthpy.plot as ep
import earthpy.spatial as es

from shapely.geometry import Polygon, mapping
import rasterio as rio
from rasterio.mask import mask
from rasterio.plot import plotting_extent
from rasterio.plot import show

import seaborn as sns

CRS = '32632'

cities = ['palermo', 'milano', 'roma', 'firenze', 'torino', 'bologna']

# Prettier plotting with seaborn
sns.set(font_scale=1.5)

city_sat_img_dict = {
"milano" : "S2B_MSIL2A_20181024T102059_N0209_R065_T32TNR_20181024T171522_RGB" + "_" + CRS + ".tiff",
"palermo" : "S2A_MSIL2A_20180712T095031_N0208_R079_T33SUC_20180712T122315_RGB" + "_" + CRS + ".tiff",
"torino": "S2B_MSIL2A_20181024T102059_N0209_R065_T32TLQ_20181024T171522_RGB" + "_" + CRS + ".tiff",
"bologna" : "S2A_MSIL2A_20170802T101031_N0205_R022_T32TPQ_20170802T101051_RGB" + "_" + CRS + ".tiff",
"firenze" : "S2B_MSIL2A_20191026T101029_N0213_R022_T32TPP_20191026T133255_RGB" + "_" + CRS + ".tiff",
"roma" : "S2A_MSIL2A_20181023T100051_N0209_R122_T32TQM_20181023T132111_RGB" + "_" + CRS + ".tiff"
}


# if you want to see crop extent
def plot_crop_extent():
	fig, ax = plt.subplots(figsize=(6, 6))
	crop_extent.plot(ax=ax)
	ax.set_title("Shapefile Crop Extent",
				 fontsize=16)

	plt.show()



# if you want to see crop extent on the raster file
def plot_crop_overlayed_on_raster():
	fig, ax = plt.subplots(figsize=(10, 8))

	ep.plot_bands(raster.read(1), cmap='terrain',
				  extent=plotting_extent(raster),
				  ax=ax,
				  title="Raster Layer with Shapefile Overlayed",
				  cbar=False)

	crop_extent.plot(ax=ax, alpha=.8)
	ax.set_axis_off()

	plt.show()

# you likely want to see the cropped raster
def plot_cropped_raster():
	# Plot your data
	ep.plot_bands(raster_crop[2],
				  extent=raster_extent,
				  cmap='Greys',
				  title="Cropped Raster Dataset",
				  scale=False)
	plt.show()

def save_cropped_raster(path_out):
	# Write data
	
	# Read each layer and write it to stack
	with rio.open(path_out, 'w', **raster_meta) as ff:
		for i in range(raster.count):
			src1 = raster_crop[i]
			ff.write_band(i+1, src1)


for city_name in cities:


	sat_img = city_sat_img_dict[city_name]

	# this is file path of the raster to be cropped
	fp = r"data/satellite_imagery/satellite/Sentinel-2_2A/" + sat_img

	# this is file path of the vector for cropping extent
	vfp = r"data/boundaries/districts/" + city_name + "/" + city_name + ".shp"


	print("*******************************************")
	print ("RASTER INFO ")
	print("*******************************************")
	raster = rio.open(fp)
	print (type(raster))
	print (raster.crs)
	"""
	The spatial extent of an Python spatial object represents the geographic 
	“edge” or location that is the furthest north, south, east and west. 
	In other words, extent represents the overall geographic coverage of the spatial object.
	"""
	print ("spatial extent", raster.bounds)
	"""
	A raster has horizontal (x and y) resolution. 
	This resolution represents the area on the ground that each pixel covers. 
	The units for your data are in meters as determined by the CRS.
	"""
	print ("resolution", raster.res)
	print (raster.meta)
	print(raster.tags(ns='IMAGE_STRUCTURE'))
	"""
	tells us the number of layers, bands
	"""
	print ("number of bands", raster.count)
	print ()

	print("*******************************************")
	print ("VECTOR INFO ")
	print("*******************************************")
	vector = gp.read_file(vfp)
	print(vector)
	print (vector.crs)
	print ()

	print ()
	vector = vector.to_crs(raster.crs)

	# our crop_extent is the vector
	crop_extent = vector

	raster_crop, raster_crop_meta = es.crop_image(raster,crop_extent)
	raster_crop_affine = raster_crop_meta["transform"]

	# Create spatial plotting extent for the cropped layer
	raster_extent = plotting_extent(raster_crop[0], raster_crop_affine)


	print (crop_extent)

	plot_crop_extent()

	plot_crop_overlayed_on_raster()

	plot_cropped_raster()


	raster_meta = raster.profile
	raster_meta.update({'transform': raster_crop_affine,
						   'height': raster_crop.shape[1],
						   'width': raster_crop.shape[2],
						   'nodata': 0.0,
						   'count':raster.count})


	img_out = sat_img.replace("_RGB.tif", "_RGB_CROPPED.tif")
	path_out = r"data/satellite_imagery/2A_source_raster/" + img_out
	save_cropped_raster(path_out)

	raster.close()

