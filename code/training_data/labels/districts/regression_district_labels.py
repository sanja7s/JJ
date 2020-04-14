#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sanja7s
reads original labels and creates a dataframe
with our id city_district created

"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

city_pro_com_dict= {
'Milano': 15146,
'Torino': 1272,
'Roma': 58091,
'Palermo': 82053,
'Bologna': 37006,
'Firenze': 48017,
}

city_region = {
'Milano': 'lombardia',
'Torino': 'piemonte',
'Roma': 'lazio_32632',
'Palermo': 'sicilia',
'Bologna': 'emilia-romagna',
'Firenze': 'toscana',	
}

SMALL_VALUE = 0.000000009



columns = ["hType_mix", "num_intersect", "bld_avg_age",\
				"LUM5_single",	"RNR_nres", "mdist_smallparks", "nig_rat_daily",\
				"nig_rat_daily3", "mdist_nres_daily", "num_community_places", \
				"num_community_places_poi", "avg_block_area", "sphi", \
				"enterprises_empl_size", "pop_rat_num", "emp_rat_num",  \
				"emp_rat_pop", "bld_rat_area", "den_nres_daily",\
				"mdist_parks", "den_nres_non-daily", "mdist_railways",\
				"mdist_highways", "mdist_water", "activity_density"]
# label_columns = ["label_"+l for l in columns]

# these are found to need np.log transformation
columns_to_log = ['hType_mix', 'mdist_nres_daily', 'mdist_smallparks', \
				'num_community_places', \
	'num_intersect',\
	'enterprises_empl_size',\
	'den_nres_daily', 'den_nres_non-daily', 'emp_rat_num', 'pop_rat_num',\
			 'emp_rat_pop',\
	"mdist_parks", "mdist_railways", "mdist_highways", "mdist_water",\
	"activity_density"]
# label_columns_to_log = ["label_"+l for l in columns_to_log]

def normalize_standardize_df(df, index_column):
	all_columns = [l for l in df.columns if l != index_column]
	saved_df = df.drop(columns=all_columns)
	all_columns = [l for l in all_columns if l != 'bld_rat_area']

	df[columns_to_log] = df[columns_to_log].apply(lambda x: np.log(x+SMALL_VALUE))
	
	standardized_features = StandardScaler().fit_transform(df[all_columns])
	minmax_features = MinMaxScaler().fit_transform(standardized_features)
	standardized_minmax_features_df = \
			pd.DataFrame(minmax_features,columns=all_columns)

	saved_df.reset_index(drop=True, inplace=True)
	standardized_minmax_features_df.reset_index(drop=True, inplace=True)
	fl = pd.concat([saved_df, standardized_minmax_features_df], axis = 1)
	return fl

def create_for_one_city(city_name = 'Roma', normalize=False):

	print ("**********************************")
	print (city_name)
	print ("**********************************")

	# base = gp.read_file("data/boundaries/districts/italy7s/"+\
	# 		city_region[city_name]+ ".shp")

	data = pd.read_csv("data/labels/merged_dataset.csv")
	data = data[data["pro_com"] == city_pro_com_dict[city_name]]
	# print(data.tail())

	print (list(sorted(data["ACE"])))

	label = data[columns + ["ACE"]].copy()

	label["city_district"] = \
		label["ACE"].astype(str).apply(lambda x: city_name.lower() + "_" + x)

	del label["ACE"]

	if normalize == True:
		# # label.set_index("city_district")
		# all_columns = [l for l in label.columns if l != 'city_district']
		# saved_label_df = label.drop(columns=all_columns)
		# all_columns = [l for l in all_columns if l != 'bld_rat_area']

		# label[columns_to_log] = label[columns_to_log].apply(lambda x: np.log(x+SMALL_VALUE))
		
		# standardized_features = StandardScaler().fit_transform(label[all_columns])
		# minmax_features = MinMaxScaler().fit_transform(standardized_features)
		# standardized_minmax_features_df = \
		# 		pd.DataFrame(minmax_features,columns=all_columns)

		# saved_label_df.reset_index(drop=True, inplace=True)
		# standardized_minmax_features_df.reset_index(drop=True, inplace=True)
		# fl = pd.concat([saved_label_df, standardized_minmax_features_df], axis = 1)
		# print (fl)
		label = normalize_standardize_df(label, "city_district")
	return label


normalize = "Total"
res = pd.DataFrame()
cities = ['Roma', 'Torino', 'Bologna', 'Firenze', 'Palermo']
for city_name in cities:
	df = create_for_one_city(city_name, normalize)
	res = pd.concat([res, df])

norm_dict = {
	True:"normalized_",
	False: "",
	"Total": "normalized_across_all_"
}
if normalize == "Total":
	res = normalize_standardize_df(res, "city_district")

# res.rename(columns={l:"label_"+l for l in label_columns})

out_dir = "preprocessed/training_data/labels/districts/"
out_file = out_dir + norm_dict[normalize] + "district_regression_labels.csv"
res.to_csv(out_file, index=0)




