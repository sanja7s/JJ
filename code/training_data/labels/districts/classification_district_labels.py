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



normalize = False
norm_dict = {
	True:"normalized_",
	False: "",
	"Total": "normalized_across_all_"
}

in_dir = "preprocessed/training_data/labels/districts/"


def regression_to_classification_labels(normalize="Total"):
	labels_file = in_dir + norm_dict[normalize] + "district_regression_labels.csv"

	reg_labels = pd.read_csv(labels_file)
	print (reg_labels.head())

	data = reg_labels.copy()

	res = data[["city_district"]].copy()
	res = res.set_index("city_district")

	for selected_var in reg_labels.columns:
		if selected_var == "city_district":
			continue
		print (selected_var)
		try:
			s = data[[selected_var, "city_district"]]
			ss = s.copy()

			# this is if we want to have 3 classes; we can try more
			ss["class_" + selected_var] = pd.qcut(s[selected_var], 3, labels=[0, 1, 2]).copy()
			# correspond to labels=["low", "medium", "high"]
			# print (s.columns)
			# print (base.columns)

			del ss[selected_var]
			ss = ss.set_index("city_district")

			res = res.join(ss)
		except Exception as e:
			print (e)

	print (res.head())
	res_file = in_dir + norm_dict[normalize] + "district_classification_labels.csv"
	res.to_csv(res_file, index=0)
			

for normalize in norm_dict.keys():
	regression_to_classification_labels(normalize)











