import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pathlib import Path
import xgboost as xgb
import numpy as np
import pickle
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.linear_model import ElasticNet, ElasticNetCV, RidgeCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# this reads in our module for selecting and 
# reading in the training data
import sys
sys.path.insert(1, 'code/training_data')
import district_training_data as dtd 


network_type = "vgg16_4096"

PCA_components = 18

LABELING_METHOD = "maximum"
AVERAGING_METHOD = "kaist"

normalize = "Total"
norm_dict = {
	True:"normalized_",
	False: "",
	"Total": "normalized_across_all_"
}

# label_columns = ["hType_mix", "num_intersect", "bld_avg_age", "emp_rat_num",\
# 				"LUM5_single",	"RNR_nres", "mdist_smallparks", "nig_rat_daily",\
# 				"nig_rat_daily3", "mdist_nres_daily", "num_community_places", \
# 				"num_community_places_poi", "avg_block_area", "sphi", \
# 				"enterprises_empl_size", "pop_rat_num",  \
# 				"emp_rat_pop", "den_nres_daily",\
# 				"mdist_parks", "den_nres_non-daily", "mdist_railways",\
# 				"mdist_highways", "mdist_water", "activity_density"]

# USE_GEO = "ALSO_GEO"
# if USE_GEO == "ALSO_GEO":
# 	features_columns = ["PCA"+str(i) for i in range(PCA_components)] + \
# 						["centroid_x", "centroid_y"]
# elif USE_GEO == "NO":
# 	features_columns = ["PCA"+str(i) for i in range(PCA_components)]
# elif USE_GEO == "ONLY_GEO":
# 	features_columns = ["centroid_x", "centroid_y"]



data = \
	dtd.get_training_data(network_type=network_type, PCA_components=PCA_components,\
		LABELING_METHOD=LABELING_METHOD, AVERAGING_METHOD=AVERAGING_METHOD,\
		pred_type='regression', normalize=normalize)

features_columns = [c for c in data.columns if c != "city_district"\
		and "label_" not in c]

# print (features_columns)

def LR(data=data, city='all', label="label_activity_density"):
	
	if city == 'all':
		data2 = data.copy()
	else:
		data2 = data[data["city_district"].str.contains(city)].copy()
	
	target = data2[["city_district", label]]
	features = data2[features_columns]
	
	X = features.values
	y = target[label].values
	
	X = sm.add_constant(X)  
	model = sm.OLS(y,X, missing='drop')	
	results = model.fit()
	
	print ("LABEL ", label)
	print_model = results.summary()
	print (print_model)
	
	return results.rsquared_adj

def LRKFold(data=data, city='all', label="label_activity_density"):
	
	if city == 'all':
		data2 = data.copy()
	else:
		data2 = data[data["city_district"].str.contains(city)].copy()
	
	target = data2[["city_district", label]]
	features = data2[features_columns]
	
	X = features.values
	y = target[label].values

	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
 
	# lm = linear_model.LinearRegression(fit_intercept=True)
	# model = lm.fit(X_train, y_train)
	# predictions = lm.predict(X_test)

	# print (model.score(X_test, y_test))


	# lm = linear_model.LinearRegression(fit_intercept=True)
	# lm = linear_model.ElasticNet()
	# clf = SVR()
	# lm = ElasticNet(alpha=0.0000001)
	# scores = cross_val_score(lm, X, y, cv=8)
	# print ("Cross-validated scores:", scores)

	clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X, y)
	print (clf.score(X, y))

	# return np.mean(scores)

def RidgeKFold(data=data, city='all', label="label_activity_density"):
	
	if city == 'all':
		data2 = data.copy()
	else:
		data2 = data[data["city_district"].str.contains(city)].copy()
	
	target = data2[["city_district", label]]
	features = data2[features_columns]
	
	X = features.values
	y = target[label].values

	clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X, y)
	print (clf.score(X, y))

	return (clf.score(X, y))

def EN(data=data, city='all', label="label_activity_density"):

	if city == 'all':
		data2 = data.copy()
	else:
		data2 = data[data["city_district"].str.contains(city)].copy()
	
	
	target = data2[["city_district", label]]
	features = data2[features_columns]
	
	X = features.values
	y = target[label].values
	
	alphas = [0.0000001, 0.0000001, 0.000001, 0.00001,\
			0.0001, 0.001, 0.01, \
			0.03, 0.05, 0.07, 0.1]

	best_res = (0,0,0,0)
	
	for a in alphas:
		model = ElasticNet(alpha=a).fit(X,y)   
		score = model.score(X, y)
		pred_y = model.predict(X)
		mse = mean_squared_error(y, pred_y)   
		print("Alpha:{0:.5f}, R2:{1:.2f}, MSE:{2:.2f}, RMSE:{3:.2f}"
		   .format(a, score, mse, np.sqrt(mse)))
		if score > best_res[1]:
			best_res = (a, score, mse, np.sqrt(mse))

	return best_res

def ENKfold(data=data, city='all', label="label_activity_density"):

	if city == 'all':
		data2 = data.copy()
	else:
		data2 = data[data["city_district"].str.contains(city)].copy()
	
	target = data2[["city_district", label]]
	features = data2[features_columns]
	
	X = features.values
	y = target[label].values
	
	alphas = [0.0000001, 0.0000001, 0.000001, 0.00001,\
			0.0001, 0.001, 0.01, \
			0.03, 0.05, 0.07, 0.1]

	elastic_cv=ElasticNetCV(alphas=alphas, cv=5)
	model = elastic_cv.fit(X, y)
	y_pred = model.predict(X)
	score = model.score(X, y)
	mse = mean_squared_error(y, y_pred)
	print("R2:{0:.3f}, MSE:{1:.2f}, RMSE:{2:.2f}"
		  .format(score, mse, np.sqrt(mse)))
			
	return ({"R2": score, "MSE": mse, \
			  "RMSE": np.sqrt(mse)})


def XGBoostKFold(data=data, city='all', label="label_activity_density"):
	
	kf = KFold(n_splits=5)
	
	if city == 'all':
		data2 = data.copy()
	else:
		data2 = data[data["city_district"].str.contains(city)].copy()
	
	target = data2[["city_district", label]]
	features = data2[features_columns]
	
	X = features.values
	y = target[label].values
 
	rmse = []
	r2 = []
	mae = []
	
	kf.get_n_splits(X, y)
	for train_index, test_index in kf.split(X, y):
		
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
	
		param_dist = {'objective' :'reg:squarederror', 'n_estimators':16}
		clf = xgb.XGBRegressor(**param_dist)
		
		clf.fit(X_train, y_train,verbose=False)
		
		predictions = clf.predict(X_test)
		rmse1 = np.sqrt(mean_squared_error(y_test, predictions))
		rmse.append(rmse1)
		
		r21 = r2_score(y_test, predictions)
		r2.append(r21)

		def OLS_R2(X, y):
			X = sm.add_constant(X)  
			model = sm.OLS(y,X, missing='drop')	
			results = model.fit()
			return results.rsquared_adj

		r22 = OLS_R2(predictions, y_test)
		# r2.append(r22)
		
		mae1 = mean_absolute_error(y_test, predictions)
		mae.append(mae1)
		
	return  ({"MAE": (np.mean(mae), np.std(mae)), \
			  "R2": (np.mean(r2), np.std(r2)), \
			  "RMSE": (np.mean(rmse), np.std(rmse))})

# res = LR()
# print (res)
# res = EN()
# print (res)

# res = ENKfold()
# print (res)

# res = XGBoostKFold()
# print (res)

# res = LRKFold()
# print (res)

res = RidgeKFold()
print (res)


def LR_out_of_city(data=data, city='all', label="label_activity_density"):
	
	data2 = data[~data["city_district"].str.contains(city)].copy()
	
	target = data2[["city_district", label]]
	features = data2[features_columns]
	
	X = features.values
	y = target[label].values
	
	X = sm.add_constant(X)  
	model = sm.OLS(y,X, missing='drop')	
	results = model.fit()

	data_city = data[data["city_district"].str.contains(city)].copy()
	
	target_city = data_city[["city_district", label]]
	features_city = data_city[features_columns]
	
	X_city = features_city.values
	y_city = target_city[label].values
	
	X_city = sm.add_constant(X_city)
	predictions = results.predict(X_city)
	r2 = r2_score(y_city, predictions)
	print (r2)
	return (r2)

def EN_out_of_city(data=data, city='all', label="label_activity_density"):

	data2 = data[~data["city_district"].str.contains(city)].copy()
	
	target = data2[["city_district", label]]
	features = data2[features_columns]
	
	X = features.values
	y = target[label].values

	data_city = data[data["city_district"].str.contains(city)].copy()
	
	target_city = data_city[["city_district", label]]
	features_city = data_city[features_columns]
	
	X_city = features_city.values
	y_city = target_city[label].values
	
	alphas = [0.0000001, 0.0000001, 0.000001, 0.00001,\
			0.0001, 0.001, 0.01, \
			0.03, 0.05, 0.07, 0.1]

	best_res = (0,0,0,0)
	best_model = None
	
	for a in alphas:
		model = ElasticNet(alpha=a).fit(X,y)   
		score = model.score(X, y)
		pred_y = model.predict(X)
		mse = mean_squared_error(y, pred_y)   
		print("Alpha:{0:.5f}, R2:{1:.2f}, MSE:{2:.2f}, RMSE:{3:.2f}"
		   .format(a, score, mse, np.sqrt(mse)))
		if score > best_res[1]:
			best_res = (a, score, mse, np.sqrt(mse))
			best_model = model

	r2 = best_model.score(X_city, y_city)
	print (r2)

	return r2

def Ridge_out_of_city(data=data, city='all', label="label_activity_density"):
	
	data2 = data[~data["city_district"].str.contains(city)].copy()

	print ("Training data lenght ", len(data2))
	
	target = data2[["city_district", label]]
	features = data2[features_columns]
	
	X = features.values
	y = target[label].values

	data_city = data[data["city_district"].str.contains(city)].copy()

	print ("Testing data lenght ", len(data_city))
	
	target_city = data_city[["city_district", label]]
	features_city = data_city[features_columns]
	
	X_city = features_city.values
	y_city = target_city[label].values

	clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X, y)
	
	r2 = clf.score(X_city, y_city)
	print (r2)

	return r2


def XGBoost_out_of_city(data=data, city='all', label="label_activity_density"):
	
	
	data2 = data[~data["city_district"].str.contains(city)].copy()

	print ("Training data lenght ", len(data2))
	
	target = data2[["city_district", label]]
	features = data2[features_columns]
	
	X = features.values
	y = target[label].values

	data_city = data[data["city_district"].str.contains(city)].copy()

	print ("Testing data lenght ", len(data_city))
	
	target_city = data_city[["city_district", label]]
	features_city = data_city[features_columns]
	
	X_city = features_city.values
	y_city = target_city[label].values
 
	param_dist = {'objective' :'reg:squarederror', 'n_estimators':16}
	clf = xgb.XGBRegressor(**param_dist)
	
	clf.fit(X, y,verbose=False)	
	predictions = clf.predict(X_city)

	rmse = np.sqrt(mean_squared_error(y_city, predictions))	
	r2 = r2_score(y_city, predictions)

	def OLS_R2(X, y):
		X = sm.add_constant(X)  
		model = sm.OLS(y,X, missing='drop')	
		results = model.fit()
		return results.rsquared_adj

	r22 = OLS_R2(predictions, y_city)
	mae = mean_absolute_error(y_city, predictions)
	print (({"R21": r2, \
			  "MAE": mae, \
			  "R22": r22, \
			  "RMSE": rmse}))
		
	return  ({"R21": r2, \
			  "MAE": mae, \
			  "R22": r22, \
			  "RMSE": rmse})


# cities = ['milano', 'bologna', 'roma', 'torino', 'firenze', 'palermo']
# for city in cities:
# 	print (city)
# 	LR_out_of_city(data=data, city=city, label="label_activity_density")

# cities = ['milano', 'bologna', 'roma', 'torino', 'firenze', 'palermo']
# for city in cities:
# 	print (city)
# 	EN_out_of_city(data=data, city=city, label="label_activity_density")

# cities = ['milano', 'bologna', 'roma', 'torino', 'firenze', 'palermo']
# for city in cities:
# 	print (city)
# 	Ridge_out_of_city(data=data, city=city, label="label_activity_density")

# cities = ['milano', 'bologna', 'roma', 'torino', 'firenze', 'palermo']
# for city in cities:
# 	print (city)
# 	XGBoost_out_of_city(data=data, city=city, label="label_activity_density")




