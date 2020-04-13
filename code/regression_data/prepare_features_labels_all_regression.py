import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json
from ast import literal_eval
import numpy as np

data_dir = "preprocessed/" 
features_in_dir = data_dir + "features/features_all/"
features_out_dir = data_dir + "regression_features/features_all/"
labels_dir = data_dir + "regression_labels/"
separate_features_dir = data_dir + "regression_features/features_separate_cities/"

network_type = "vgg16_4096" 

PCA_components = 12

SMALL_VALUE = 0.00009
label_columns = ["hType_mix", "num_intersect", "bld_avg_age", "emp_rat_num",\
				"LUM5_single",	"RNR_nres", "mdist_smallparks", "nig_rat_daily",\
				"nig_rat_daily3", "mdist_nres_daily", "num_community_places", \
				"num_community_places_poi", "avg_block_area", "sphi", \
				"enterprises_empl_size", "pop_rat_num",  \
				"emp_rat_pop", "bld_rat_area", "den_nres_daily",\
				"mdist_parks", "den_nres_non-daily", "mdist_railways",\
				"mdist_highways", "mdist_water", "activity_density"]
				
columns_to_log = ['hType_mix', 'mdist_nres_daily', 'mdist_smallparks', 'num_community_places', \
	'num_intersect',\
	'enterprises_empl_size',\
	'den_nres_daily', 'den_nres_non-daily', 'emp_rat_num', 'pop_rat_num', 'emp_rat_pop',\
	"mdist_parks", "mdist_railways", "mdist_highways", "mdist_water",\
	"activity_density"]

label_columns_to_log = ["label_"+l for l in columns_to_log]

if network_type == "vgg19":
	features_file = "Italy_6_cities_vgg19_pca"+str(PCA_components)+"_linear_fc_thirdlast_layer.csv"
elif network_type == "resnet50":
	features_file = "Italy_6_cities_resnet_pca"+str(PCA_components)+"_second_last_layer.csv"
elif network_type == "vgg16_4096":
	features_file = "Italy_6_cities_resnet_pca" + str(PCA_components) + "_vgg16_4096.csv"
geo_features_file = "all_imagelets_geocoord.csv"

def clean_labels_features(city_name):
	
	data = pd.read_csv(labels_dir + city_name + "_imagelet_labels.csv")
	if city_name in ['bologna', 'firenze']:
		del data["label_bld_rat_area"]
	data2 = data[data != -1]
	data3 = data2.dropna()
	# print (data3)
	data3 = data3.rename(columns={"imagelet":"imageName"})
	data3["imageName"] = data3["imageName"].apply(lambda x: x.replace(".shp", ""))
	data3 = data3.drop(columns= [ c for c in data3.columns if "Unnamed" in c])
	data3.to_csv(labels_dir + city_name + "_imagelet_labels_clean.csv", index=0)
	
	data1 = pd.read_csv(features_in_dir + features_file)
	data1["imageName"] = data1["imageName"].apply(lambda x: x.replace(".jpg", ""))
	data4 = pd.merge(data1,data3, on="imageName", how="inner")
	for l in label_columns:
		try:
			data4 =  data4.drop(columns=["label_" + l])
		except:
			print ("could not drop column ", l)
			pass
	data4 = data4.drop(columns= [ c for c in data4.columns if "Unnamed" in c])
	data5 = pd.read_csv(features_in_dir + geo_features_file)
	data5["centroid_x"] = data5["centroid"].apply(lambda c: literal_eval(c)[0])
	data5["centroid_y"] = data5["centroid"].apply(lambda c: literal_eval(c)[1])
	del data5["centroid"]
	del data5["Unnamed: 0"]
	data6 = pd.merge(data4,data5, on="imageName", how="inner")
	data6.to_csv(features_out_dir + features_file.replace(".csv", "_") + \
		 city_name + "_clean.csv", index=0)



def create_normalized_labels_features(city_name):

	clean_labels_features(city_name)
	
	labels = pd.read_csv(labels_dir + city_name + "_imagelet_labels_clean.csv")
	features = pd.read_csv(features_out_dir + \
			features_file.replace(".csv", "_") + city_name + "_clean.csv")
	# print (features)
	data = pd.merge(features,labels, on="imageName", how="inner")

	data2 = data.set_index("imageName")

	data2[label_columns_to_log] = data2[label_columns_to_log].apply(lambda x: np.log(x+SMALL_VALUE))
	all_columns = data2.columns
	standardized_features = StandardScaler().fit_transform(data2)
	minmax_features = MinMaxScaler().fit_transform(standardized_features)
	standardized_minmax_features_df = \
			pd.DataFrame(minmax_features,columns=all_columns)
	fl = pd.concat([features["imageName"], standardized_minmax_features_df], axis = 1)
	
	fl["city"] = city_name

	fl = fl.reset_index()
	fl.to_csv(separate_features_dir + \
	features_file.replace(".csv", "_" + city_name + "_labels_features.csv"), index=0)

	return fl

cities = ['palermo', 'torino', 'bologna','firenze', 'roma']

d1 = create_normalized_labels_features(city_name="milano")
print (d1)
for city_name in cities:
	print (city_name)
	d2 = create_normalized_labels_features(city_name)
	# print (d2)
	d1 = pd.concat([d1, d2], ignore_index=True, sort=False)
	print ('done')
d1.to_csv(features_out_dir + \
	features_file.replace(".csv","_labels_features.csv"), index=0)


