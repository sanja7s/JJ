import pandas as pd
import geopandas as gp
import os
from shutil import copy2
from pathlib import Path

THRESHOLD = .5

CRS = '32632'

LABELING_METHOD = "maximum"

city_sat_img_dict = {
"milano" : "S2B_MSIL2A_20181024T102059_N0209_R065_T32TNR_20181024T171522_RGB" + "_" + CRS,
"palermo" : "S2A_MSIL2A_20180712T095031_N0208_R079_T33SUC_20180712T122315_RGB" + "_" + CRS,
"torino": "S2B_MSIL2A_20181024T102059_N0209_R065_T32TLQ_20181024T171522_RGB" + "_" + CRS,
"bologna" : "S2A_MSIL2A_20170802T101031_N0205_R022_T32TPQ_20170802T101051_RGB" + "_" + CRS,
"firenze" : "S2B_MSIL2A_20191026T101029_N0213_R022_T32TPP_20191026T133255_RGB" + "_" + CRS,
"roma" : "S2A_MSIL2A_20181023T100051_N0209_R122_T32TQM_20181023T132111_RGB" + "_" + CRS
}


def extract_imagelet_label(city_name, file_path, file_name, label):

	image_shape = gp.read_file(file_path)
	image_potential_labels = gp.overlay(image_shape, label, how='intersection')

	if image_potential_labels.empty:
		# print ("was empty")
		return {"imagelet":file_name, "label_district":-1}

	image_potential_labels["area"] = image_potential_labels.area

	if LABELING_METHOD == "threshold":
		image_potential_labels["label?"] = \
			 (image_potential_labels["area"]/image_shape.area.values[0] > THRESHOLD)
		try:
			selected_label = \
				image_potential_labels[image_potential_labels["label?"]]["district"].values[0]
		except IndexError as i:
			# print (i)
			return {"imagelet":file_name, "label_district":-1}

	elif LABELING_METHOD == "maximum":
		try:
			selected_label = \
				image_potential_labels.loc[image_potential_labels['area']\
					.idxmax()]["district"]
		except Exception as e:
			print (e)
			return {"imagelet":file_name, "label_district":-1}


	one_label = {"imagelet":file_name, "label_district":selected_label}
	return one_label

def batch_extract_labels(city_name, in_dir):
	label_lst = []
	print('LABELLING  ')

	try:
		label = \
			gp.read_file(shapes_dir + city_name + "_epsg_" +CRS+ ".shp")
	except:
		return None

	max_label = label["district"].max()
	min_label = label["district"].min()
	print (min_label, max_label)

	i = 0

	for root, subdirs, fl in os.walk(in_dir):
		for f in fl:
			if f.endswith(".shp"):
				#print (f)
				fPath = os.path.join(root, f)
				# print('LABELLING image ', f)
				one_label = extract_imagelet_label(city_name, fPath, f, label) 
				label_lst.append(one_label)

				# i+=1
				# if i > 10:
				# 	label_df = pd.DataFrame(label_lst)
				# 	return label_df


	label_df = pd.DataFrame(label_lst)
	# print (label_df)
	return label_df

def label_one_city(city_name = "milano"):
	in_dir = "data/satellite_imagery/2A_imagelet_shapes/" \
		+ city_sat_img_dict[city_name]
	out_file = "preprocessed/training_data/district_imagelets_link/" + \
		city_name.lower() + "_imagelet_labels_"+LABELING_METHOD+".csv"

	try: 
		labels = batch_extract_labels(city_name, in_dir)
	except Exception as e:
		print (e)

	# labels.to_csv(out_file)
	return labels

shapes_dir = "data/boundaries/districts/"
res_dir = "preprocessed/training_data/district_imagelets_link/" \
		+ "all_imagelet_labels_"+LABELING_METHOD+".csv"

# these shapes are created by us by dissolving the block shapefiles
cities = ['bologna', 'firenze', 'torino', 'palermo', 'roma', 'milano'] 
res = pd.DataFrame()
for city_name in cities:
	df_one = label_one_city(city_name)
	df_one["city_district"] = \
		df_one["label_district"].astype(str)\
		.apply(lambda x: city_name + "_" + x)
	res = pd.concat([res, df_one])
res.to_csv(res_dir, index=0)

# label_one_city()




