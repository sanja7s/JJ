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
'Roma': 'lazio_32632',
'Palermo': 'sicilia',
'Bologna': 'emilia-romagna',
'Firenze': 'toscana',	
}


def create_for_one_city(city_name = 'Roma'):

	print ("**********************************")
	print (city_name)
	print ("**********************************")

	base = gp.read_file("data/boundaries/districts/italy7s/"+\
			city_region[city_name]+ ".shp")
	data = pd.read_csv("data/labels/merged_dataset.csv")

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


	try:
		s = data[["ACE"] + label_columns]

		ss = s.copy()
		ss["district"] = s["ACE"].copy()

		label = base.merge(ss, on="ACE")
		label['geometry'] = label.buffer(0.01) 
		label2 = label.dissolve(by="district", aggfunc='mean')
		label2 = label2.reset_index()

		label2.to_file("preprocessed/district_labels/label_shapes/"\
			 +city_name.lower()+"_epsg_32632.shp")
	except Exception as e:
		print ("Could not save! ", e)
		
	

# cities = ['Milano', 'Torino', 'Bologna', 'Firenze', 'Palermo', 'Roma']
# for city_name in cities:
# 	create_for_one_city(city_name)

create_for_one_city()




