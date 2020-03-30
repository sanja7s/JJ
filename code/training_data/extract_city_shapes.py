#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sanja7s


district data from 
https://www.istat.it/it/archivio/104317#accordions

prettify this code 

"""

import pandas as pd
import json
import geopandas as gp
import numpy as np


city_name = 'Firenze'

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

base = gp.read_file("data/boundaries/districts/italy7s/"+\
		city_region[city_name]+ ".shp")

base = base[base["PRO_COM"] == city_pro_com_dict[city_name]]

print(base.columns)
print(base.head())

base.to_file("data/boundaries/districts/"+ city_name.lower()+\
		"/" +city_name.lower()+ ".shp")
	









