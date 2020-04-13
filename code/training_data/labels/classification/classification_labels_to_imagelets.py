import pandas as pd
import geopandas as gp
import os
from shutil import copy2
from pathlib import Path

THRESHOLD = .5

CRS = '32632'

city_sat_img_dict = {
"milano" : "S2B_MSIL2A_20181024T102059_N0209_R065_T32TNR_20181024T171522_RGB" + "_" + CRS,
"palermo" : "S2A_MSIL2A_20180712T095031_N0208_R079_T33SUC_20180712T122315_RGB" + "_" + CRS,
"torino": "S2B_MSIL2A_20181024T102059_N0209_R065_T32TLQ_20181024T171522_RGB" + "_" + CRS,
"bologna" : "S2A_MSIL2A_20170802T101031_N0205_R022_T32TPQ_20170802T101051_RGB" + "_" + CRS,
"firenze" : "S2B_MSIL2A_20191026T101029_N0213_R022_T32TPP_20191026T133255_RGB" + "_" + CRS,
"roma" : "S2A_MSIL2A_20181023T100051_N0209_R122_T32TQM_20181023T132111_RGB" + "_" + CRS
}


# "bld_avg_age" is called first alone so excluded from the list
label_columns = ["hType_mix", "num_intersect",\
				"LUM5_single",	"RNR_nres", "mdist_smallparks", "nig_rat_daily",\
				"nig_rat_daily3", "mdist_nres_daily", "num_community_places", \
				"num_community_places_poi", "avg_block_area", "sphi", \
				"enterprises_empl_size", "pop_rat_num", "emp_rat_num",  \
				"emp_rat_pop", "bld_rat_area", "den_nres_daily",\
				"mdist_parks", "den_nres_non-daily", "mdist_railways",\
				"mdist_highways", "mdist_water", "activity_density"]


def extract_imagelet_label(city_name, file_path, file_name, selected_var, label):

	image_shape = gp.read_file(file_path)
	image_potential_labels = gp.overlay(image_shape, label, how='intersection')

	if image_potential_labels.empty:
		# print ("was empty")
		return {"imagelet":file_name, "label_"+selected_var:-1}

	image_potential_labels["area"] = image_potential_labels.area
	image_potential_labels["label?"] = \
		 (image_potential_labels["area"]/image_shape.area.values[0] > THRESHOLD)
	
	# print (image_potential_labels["area"])
	# print (label.crs, image_shape.crs)
	# print (image_potential_labels["label?"])
	# print(image_potential_labels[image_potential_labels["label?"]]["class"])

	try:
		selected_label = \
			image_potential_labels[image_potential_labels["label?"]]["class"].values[0]
	except IndexError as i:
		print (i)
		return {"imagelet":file_name, "label_"+selected_var:-1}

	# print (image_potential_labels["area"])
	# print (label.crs, image_shape.crs)
	# print (image_potential_labels["label?"])
	print(image_potential_labels[image_potential_labels["label?"]]["class"].values[0])

	one_label = {"imagelet":file_name, "label_"+selected_var:selected_label}
	return one_label

def batch_extract_labels(city_name, in_dir, selected_var, \
						create_high_low_examples=False, examples_folder=None,\
						jpg_dir=None):
	label_lst = []
	print('LABELLING  ', selected_var)

	try:
		label = \
			gp.read_file("preprocessed/labels/label_shapes/"+\
				city_name+"/"+selected_var+"_epsg_32632.shp")
	except:
		return None


	if create_high_low_examples:
		label_examples_folder = examples_folder+selected_var+"/"
		Path(label_examples_folder).mkdir(parents=True, exist_ok=True)
		high_examples_folder = label_examples_folder + "high/"
		low_examples_folder = label_examples_folder + "low/"
		Path(high_examples_folder).mkdir(parents=True, exist_ok=True)
		Path(low_examples_folder).mkdir(parents=True, exist_ok=True)

	for root, subdirs, fl in os.walk(in_dir):
		for f in fl:
			if f.endswith(".shp"):
				#print (f)
				fPath = os.path.join(root, f)
				# print('LABELLING image ', f)
				one_label = extract_imagelet_label(city_name, fPath, f, selected_var, label) 
				label_lst.append(one_label)

				if create_high_low_examples:
					if one_label["label_" + selected_var] in [1,-1]:
						continue
					jName = f.replace(".shp", ".jpg")
					jPath = jpg_dir
					src = os.path.join(jPath, jName)
					if one_label["label_" + selected_var] == 2:
						dst = os.path.join(high_examples_folder, jName)
					elif one_label["label_" + selected_var] == 0:
						dst = os.path.join(low_examples_folder, jName)

					copy2(src, dst)


	label_df = pd.DataFrame(label_lst)
	print (label_df)
	return label_df


	

def label_one_city(city_name = "milano"):
	in_dir = "preprocessed/training_images/2A_imagelet_shapes/" + city_sat_img_dict[city_name]
	out_file = "preprocessed/labels/" + city_name.lower() +"_imagelet_labels.csv"

	examples_folder = "preprocessed/high_low_examples/"+city_name+"/"
	jpg_dir = \
		"preprocessed/training_images/2A_imagelet_jpgs/training_6_cities/" \
		+ city_sat_img_dict[city_name]

	try: 
		labels = batch_extract_labels(city_name, in_dir, "bld_avg_age",True,examples_folder,jpg_dir)
	except Exception as e:
		print (e)

	for selected_var in label_columns:
		one_res = batch_extract_labels(city_name, in_dir,selected_var,True,examples_folder,jpg_dir)
		if one_res is not None:
			labels = pd.merge(labels, one_res, on="imagelet")
	labels.to_csv(out_file)



# cities = ['bologna', 'firenze'] # 'torino', 'palermo', 'roma', 'milano', 
# for city_name in cities:
# 	label_one_city(city_name)


label_one_city()




