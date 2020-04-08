import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pathlib import Path
import xgboost as xgb
import numpy as np
import pickle


data_dir = "preprocessed/" 
features_dir = data_dir + "regression_features/features_all/"
separate_features_dir = data_dir + "regression_features/features_separate_cities/"
model_dir = "resources/regression/" 

PCA_components = 64

USE_GEO = "GEO"
if USE_GEO == "GEO":
	features_columns = ["PCA"+str(i) for i in range(PCA_components)] + \
						["centroid_x", "centroid_y"]
else:
	features_columns = ["PCA"+str(i) for i in range(PCA_components)]
		
network_type = "vgg16_4096"

label_columns = ["hType_mix", "num_intersect", "bld_avg_age", "emp_rat_num",\
				"LUM5_single",	"RNR_nres", "mdist_smallparks", "nig_rat_daily",\
				"nig_rat_daily3", "mdist_nres_daily", "num_community_places", \
				"num_community_places_poi", "avg_block_area", "sphi", \
				"enterprises_empl_size", "pop_rat_num",  \
				"emp_rat_pop", "den_nres_daily",\
				"mdist_parks", "den_nres_non-daily", "mdist_railways",\
				"mdist_highways", "mdist_water", "activity_density"]

if network_type == "vgg19":
	features_file = "Italy_6_cities_vgg19_pca"+str(PCA_components)+"_linear_fc_thirdlast_layer.csv"
elif network_type == "resnet50":
	features_file = "Italy_6_cities_resnet_pca"+str(PCA_components)+"_second_last_layer.csv"
elif network_type == "vgg16_4096":
	features_file = "Italy_6_cities_resnet_pca" + str(PCA_components) + "_vgg16_4096.csv"



def get_normalized_labels_features_city(city_name):

	df = pd.read_csv(separate_features_dir + \
		features_file.replace(".csv", "_" + city_name + "_labels_features.csv"))

	df["city_image"] = df.\
		apply(lambda x: x.city + "_" + x.imageName, axis = 1)

	print ("ONLY city ", len(df))
	
	del df['imageName']
	del df['city']
	del df['index']
	return df


def get_normalized_labels_features_out_of_city(city_name):

	df = pd.read_csv(features_dir + \
		features_file.replace(".csv", "_labels_features.csv"))

	print ("ALL cities ", len(df))

	df = df[df["city"] != city_name]

	print ("OTHER cities ", len(df))

	df["city_image"] = df.\
		apply(lambda x: x.city + "_" + x.imageName, axis = 1)
	
	del df['imageName']
	del df['city']
	del df['index']
	return df


def train_label_i(data, city_name, label="label_activity_density"):
	
	data2 = data.copy()
	
	target = data2[["city_image", label]]
	features = data2[features_columns]
	
	X = features.values
	y = target[label].values
	
	param_dist = {'objective' :'reg:squarederror', 'n_estimators':16}
	clf = xgb.XGBRegressor(**param_dist)

	clf.fit(X, y,verbose=False)
	
	pickle.dump(clf, open(model_dir + "out_of_" + city_name \
		+ "_" + label + '_all_reg.dat' , "wb"))
	
	# print(clf.score(X, y))

def infer_label_i(data, city_name, label="label_hType_mix"):
	
	data2 = data.copy()
	target = data2[["city_image", label]]
	features = data2[features_columns]
	
	X = features.values
	y = target[label].values
	
	clf = pickle.load(open(model_dir + "out_of_" +city_name \
		+ "_" + label + '_all_reg.dat', 'rb'))
	
	predictions = clf.predict(X) 
	
	rmse = np.sqrt(mean_squared_error(y, predictions))
	r2 = r2_score(y, predictions)
	mae = mean_absolute_error(y, predictions)
	   
	return  ({"MAE": mae, "R2": r2, "RMSE": rmse})

def train_out_of(city_name):

	train_data = get_normalized_labels_features_out_of_city(city_name)
	print ("Training out of ", city_name)
	for col in label_columns:
		label = "label_" + col
		# print (col)
		train_label_i(train_data, city_name, label)


def predict(city_name):

	SCORES = {}
	print ("Testing ", city_name)
	test_data = get_normalized_labels_features_city(city_name)

	for col in label_columns:
		label = "label_" + col
		# print (col)
		res = infer_label_i(test_data, city_name, label)
		SCORES[label] = res

	res = pd.DataFrame(SCORES)
	res.to_csv("results/regression/unseen_city/" + city_name + "_inference.csv")


cities = ['bologna', 'firenze', 'palermo', 'torino', 'roma', 'milano']
for city_name in cities:
	train_out_of(city_name)
	predict(city_name)



