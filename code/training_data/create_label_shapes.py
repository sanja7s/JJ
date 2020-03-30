#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sanja7s


"""

import pandas as pd
import json
import geopandas as gp
import numpy as np




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
'Roma': 'lazio',
'Palermo': 'sicilia',
'Bologna': 'emilia-romagna',
'Firenze': 'toscana',	
}


def create_for_one_city(city_name = 'Torino'):

	print ("**********************************")
	print (city_name)
	print ("**********************************")

	base = gp.read_file("data/boundaries/districts/italy7s/"+\
			city_region[city_name]+ ".shp")
	data = pd.read_csv("data/training_data/labels/merged_dataset.csv")

	base = base[base["PRO_COM"] == city_pro_com_dict[city_name]]

	# print(base.columns)
	# print(base.head())
	# print(data.columns)
	# print(data.tail())

	data = data[data["pro_com"] == city_pro_com_dict[city_name]]
	# print(data.tail())

	print (list(sorted(data["ACE"])))

	label_columns = ["hType_mix", "num_intersect", "bld_avg_age",\
					"LUM5_single",	"RNR_nres", "mdist_smallparks", "nig_rat_daily",\
					"nig_rat_daily3", "mdist_nres_daily", "num_community_places", \
					"num_community_places_poi", "avg_block_area", "sphi", \
					"enterprises_empl_size", "pop_rat_num", "emp_rat_num",  \
					"emp_rat_pop", "bld_rat_area", "den_nres_daily",\
					"mdist_parks", "den_nres_non-daily", "mdist_railways",\
					"mdist_highways", "mdist_water", "activity_density"]



	for selected_var in label_columns:
		print (selected_var)
		try:
			s = data[[selected_var, "ACE"]]

			ss = s.copy()
			# this is if we want to have 3 classes; we can try more
			ss["class"] = pd.qcut(s[selected_var], 3, labels=[0, 1, 2]).copy()
			# correspond to labels=["low", "medium", "high"]
			# print (s.columns)
			# print (base.columns)

			label = base.merge(ss, on="ACE")
			label = label.drop(columns=["ACE"])
			label2 = label.dissolve(by="class", aggfunc='mean')
			label2 = label2.reset_index()
			# print (label2[["class", selected_var]])
			label2["class"] = label2["class"].astype(int)

			label2.to_file("preprocessed/labels/label_shapes/" +city_name.lower()+"/"+selected_var+"_epsg_32632.shp")
		except Exception as e:
			print ("Could not save! ", e)
			continue
	

# cities = ['Torino', 'Milano', 'Bologna', 'Firenze', 'Palermo', 'Roma']
# for city_name in cities:
# 	create_for_one_city(city_name)

create_for_one_city()




