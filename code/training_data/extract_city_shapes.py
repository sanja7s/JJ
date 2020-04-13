#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sanja7s


district data from 
https://www.istat.it/it/archivio/104317#accordions

prettify this code 

"""

# import pandas as pd
import geopandas as gp



city_name = 'Roma'

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

if city_name == 'Roma':
	base = gp.read_file("data/boundaries/districts/italy7s/"+\
		city_region[city_name]+ "_32632.shp")
else:
	base = gp.read_file("data/boundaries/districts/italy7s/"+\
			city_region[city_name]+ ".shp")
print(base.columns)
print(base.head())
print (base.crs)


base = base[base["PRO_COM"] == city_pro_com_dict[city_name]]

# base.crs = {'init' :'epsg:32632'}
# print (base.crs)
# base = base.to_crs(base.crs)
# print (base.crs)




base.to_file("data/boundaries/districts/"+ city_name.lower()+\
		"/" +city_name.lower()+ ".shp")
	









