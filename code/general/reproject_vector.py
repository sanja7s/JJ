import geopandas as gp


CRS = '32632'
new_crs='EPSG:' + CRS


def reproject_shape(input_path, to_crs = new_crs):

	base = gp.read_file(input_path)
	print (base.crs)
	base = base.to_crs(to_crs)
	# Output path
	outfp = input_path.replace(".shp", "_" +CRS+".shp")
	print (outfp)
	# Save to disk
	base.to_file(outfp)


# for selected_var in \
# 	["hType_mix", "num_intersect", "bld_avg_age", "emp_rat_num"]:
# 	reproject_shape("preprocessed/training/labels/milano/" +selected_var +".shp")




# reproject_shape("data/boundaries/districts/" +city_name.lower() +"/" \
# 		 +city_name.lower() + ".shp")


reproject_shape("data/boundaries/districts/italy7s/lazio.shp")