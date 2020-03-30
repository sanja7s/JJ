import os
import numpy as np
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import earthpy as et
from osgeo import gdal


CRS = '32632'
new_crs='EPSG:' + CRS

cities = ['milano', 'palermo'] # , 'bologna'



city_sat_img_dict = {
"milano" : "S2B_MSIL2A_20181024T102059_N0209_R065_T32TNR_20181024T171522_RGB.tif",
"palermo" : "S2A_MSIL2A_20180712T095031_N0208_R079_T33SUC_20180712T122315_RGB.tif",
"bologna": "S2A_MSIL2A_20170802T101031_N0205_R022_T32TPQ_20170802T101051_RGB.tif",
"firenze" : "S2B_MSIL2A_20191026T101029_N0213_R022_T32TPP_20191026T133255_RGB.tif",
"roma" : "S2A_MSIL2A_20181023T100051_N0209_R122_T32TQM_20181023T132111_RGB.tif",
"torino" : "S2B_MSIL2A_20181024T102059_N0209_R065_T32TLQ_20181024T171522_RGB.tif"
}


def reproject_et(inpath, outpath, new_crs):
	dst_crs = new_crs
	with rio.open(inpath) as src:
		transform, width, height = calculate_default_transform(
					src.crs, dst_crs, src.width, src.height, *src.bounds)
		kwargs = src.meta.copy()
		kwargs.update({
			'crs': dst_crs,
			'transform': transform,
			'width': width,
			'height': height
		})

		with rio.open(outpath, 'w', **kwargs) as dst:
			for i in range(1, src.count + 1):
				reproject(
					source=rio.band(src, i),
					destination=rio.band(dst, i),
					src_transform=src.transform,
					src_crs=src.crs,
					dst_transform=transform,
					dst_crs=dst_crs,
					resampling=Resampling.nearest)


def reproject_gdal(inpath, outpath, new_crs):

	gdal.Warp(outpath,inpath,dstSRS=new_crs)



for city_name in ["torino"]:

	inputfile = city_sat_img_dict[city_name]
	outputfile = inputfile.replace(".tif", "_" + CRS + ".tiff")

	inpath = "data/training_data/satellite/Sentinel-2_2A/" + inputfile
	outpath = "data/training_data/satellite/Sentinel-2_2A/" + outputfile

	# reproject_et(inpath, outpath)

	# preferred method
	reproject_gdal(inpath, outpath, new_crs)


