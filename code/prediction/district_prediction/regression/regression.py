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
from sklearn.linear_model import ElasticNet, ElasticNetCV, RidgeCV, LinearRegression, LassoCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

import time
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_validate, cross_val_predict

# this reads in our module for selecting and 
# reading in the training data
import sys
sys.path.insert(1, 'code/training_data')
import district_training_data as dtd 


network_type = "vgg16_4096"

PCA_components = 16

LABELING_METHOD = "maximum"
AVERAGING_METHOD = "kaist"

normalize = True
norm_dict = {
	True:"normalized_",
	False: "",
	"Total": "normalized_across_all_"
}


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

label_columns = [l for l in data.columns if "label_" in l]

def plot_regression_results(ax, y_true, y_pred, title, scores, elapsed_time):
	"""Scatter plot of the predicted vs true targets."""
	ax.plot([y_true.min(), y_true.max()],
			[y_true.min(), y_true.max()],
			'--r', linewidth=2)
	ax.scatter(y_true, y_pred, alpha=0.2)

	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	ax.spines['left'].set_position(('outward', 10))
	ax.spines['bottom'].set_position(('outward', 10))
	ax.set_xlim([y_true.min(), y_true.max()])
	ax.set_ylim([y_true.min(), y_true.max()])
	ax.set_xlabel('Actual')
	ax.set_ylabel('Predicted')
	extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
						  edgecolor='none', linewidth=0)
	ax.legend([extra], [scores], loc='lower right')
	# title = title + '\n Evaluation in {:.2f} seconds'.format(elapsed_time)
	ax.set_title(title)

def plot_different_estimators(data=data, city='all', label="label_activity_density"):

	params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
		  'learning_rate': 0.01, 'loss': 'ls'}

	estimators = [
	('LinearRegression', LinearRegression()),
	('ElasticNet', ElasticNetCV()),
	('SVM regressor', SVR()),
	('Ridge', RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])),
	('Random Forest', RandomForestRegressor(random_state=42)),
	('XGBoost', GradientBoostingRegressor(**params))
	]	

	estimators_subset = [
	('LinearRegression', LinearRegression()),
	('ElasticNet', ElasticNetCV()),
	('SVM regressor', SVR()),
	('XGBoost', GradientBoostingRegressor(**params))
	]	

	if city == 'all':
		data2 = data.copy()
	else:
		data2 = data[data["city_district"].str.contains(city)].copy()
	
	target = data2[["city_district", label]]
	features = data2[features_columns]
	
	X = features.values
	y = target[label].values

	fig, axs = plt.subplots(2, 2, figsize=(10, 10))
	axs = np.ravel(axs)

	for ax, (name, est) in zip(axs, estimators_subset):
		start_time = time.time()
		score = cross_validate(est, X, y,
							   scoring=['r2', 'neg_mean_absolute_error'],
							   n_jobs=-1, verbose=0, cv=5)
		elapsed_time = time.time() - start_time

		y_pred = cross_val_predict(est, X, y, n_jobs=-1, verbose=0, cv=5)
		plot_regression_results(
			ax, y, y_pred,
			name,
			(r'$R^2={:.2f} \pm {:.2f}$' + '\n' + r'$MAE={:.2f} \pm {:.2f}$')
			.format(np.mean(score['test_r2']),
					np.std(score['test_r2']),
					-np.mean(score['test_neg_mean_absolute_error']),
					np.std(score['test_neg_mean_absolute_error'])),
			elapsed_time)

	# plt.suptitle('Single predictors versus stacked predictors')
	plt.tight_layout()
	plt.subplots_adjust(top=0.9)
	plt.savefig("results/prediction/districts/regression/different_estimators_"+\
		label + ".png", dpi=100)
	# plt.show()

def regress_variables(data=data, city='all', label="label_activity_density"):

	land_use = [
	"LUM5_single","RNR_nres","mdist_smallparks",
	"hType_mix", "nig_rat_daily", "mdist_nres_daily",
	"num_community_places", "num_community_places_poi"]
	land_use_cols = ["label_"+l for l in land_use]

	small_blocks = [
	"avg_block_area","num_intersect", "sphi"]
	small_blocks_cols = ["label_"+l for l in small_blocks]

	params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
		  'learning_rate': 0.01, 'loss': 'ls'}

	estimators = [
	('LinearRegression', LinearRegression(fit_intercept=True)),
	('ElasticNet', ElasticNetCV(fit_intercept=True)),
	('SVM regressor', SVR()),
	('Ridge', RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])),
	('Random Forest', RandomForestRegressor(random_state=42)),
	('XGBoost', GradientBoostingRegressor(**params))
	]	

	# est = SVR() #GradientBoostingRegressor(**params)

	if city == 'all':
		data2 = data.copy()
	else:
		data2 = data[data["city_district"].str.contains(city)].copy()


	

	for (est_name, est) in estimators:

		res = {"variable":[], "R2":[], "R2std":[],  "mae":[], "maestd":[]}

		for label in land_use_cols + small_blocks_cols + ['label_pop_rat_num']: #label_columns:

			target = data2[["city_district", label]].copy()
			features = data2[features_columns].copy()
			
			X = features.values
			y = target[label].values

			score = cross_validate(est, X, y,
				scoring=['r2', 'neg_mean_absolute_error'],
				n_jobs=-1, verbose=0, cv=5)


			print(np.mean(score['test_r2']),
				np.std(score['test_r2']),
				-np.mean(score['test_neg_mean_absolute_error']),
				np.std(score['test_neg_mean_absolute_error']))

			res["variable"].append(label)
			res["R2"].append(np.mean(score['test_r2']))
			res["R2std"].append(np.std(score['test_r2']))
			res["mae"].append(-np.mean(score['test_neg_mean_absolute_error']))
			res["maestd"].append(np.std(score['test_neg_mean_absolute_error']))
		res = pd.DataFrame(res)
		res.to_csv("results/prediction/districts/regression/two_cat_variables"+est_name+".csv", index=0)
		res.to_latex("results/prediction/districts/regression/two_cat_variables"+est_name+".tex",\
			float_format="{:0.3f}".format)
	return res


def regress_vitality_from_variables(data=data, city='all', label="label_activity_density"):

	land_use = [
	"LUM5_single","RNR_nres","mdist_smallparks",
	"hType_mix", "nig_rat_daily", "mdist_nres_daily",
	"num_community_places", "num_community_places_poi"]
	land_use_cols = ["label_"+l for l in land_use]

	small_blocks = [
	"avg_block_area","num_intersect", "sphi"]
	small_blocks_cols = ["label_"+l for l in small_blocks]

	params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
		  'learning_rate': 0.01, 'loss': 'ls'}

	estimators = [
	('LinearRegression', LinearRegression(fit_intercept=True)),
	('ElasticNet', ElasticNetCV(fit_intercept=True)),
	('SVM regressor', SVR()),
	('Ridge', RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])),
	('Random Forest', RandomForestRegressor(random_state=42)),
	('XGBoost', GradientBoostingRegressor(**params))
	]	

	selected_estimators = [
	('SVM regressor', SVR()),
	# ('XGBoost', GradientBoostingRegressor(**params))
	]

	selected_estimators_vitality = [
	('XGBoost', GradientBoostingRegressor(**params))
	]
	# est = SVR() #GradientBoostingRegressor(**params)

	if city == 'all':
		data2 = data.copy()
	else:
		data2 = data[data["city_district"].str.contains(city)].copy()

	target = data2[["city_district"] + land_use_cols + small_blocks_cols + ["label_activity_density"]].copy()
	features = data2[features_columns].copy()
	
	X = features.values
	y = target[land_use_cols + small_blocks_cols + ["label_activity_density"]].values

	print ("X", X.shape)
	print ("y", y.shape)

	FI_df = pd.DataFrame()

	for rnd_st in range(10):

		X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size=0.33, random_state=rnd_st)

		w, h = 11, 123;
		# RES = np.array([[0 for x in range(w)] for y in range(h)])
		# RES = np.zeros((h,w))
		RES = pd.DataFrame()
		new_y = y_test[:,-1]

		# print (y_train)
		print ("y train", y_train.shape)
		print ("y test", y_test.shape)

		for (est_name, est) in selected_estimators:

			res = {"variable":[], "R2":[], "R2std":[],  "mae":[], "maestd":[]}
			i = 0
			cnt = 0

			for label in land_use_cols + small_blocks_cols : #label_columns: + ['label_pop_rat_num']

				# target = data2[["city_district", label]].copy()
				# features = data2[features_columns].copy()
				y_train_small = y_train[:,i]
				y_test_small = y_test[:,i]

				# print (y_train_small)
				print ("y_train_small ", y_train_small.shape)

				# # score = cross_validate(est, X, y,
				# # 	scoring=['r2', 'neg_mean_absolute_error'],
				# # 	n_jobs=-1, verbose=0, cv=5)

				# X_train, X_test, y_train, y_test = train_test_split(
				# 	X, y, test_size=0.33, random_state=42)

				s = est.fit(X_train, y_train_small)
				score = s.score(X_test, y_test_small)
				y_pred = est.predict(X_test)

				# print ("y_pred ", y_pred)

				r2 = r2_score(y_test_small, y_pred)

				if True: #r2 > .2:
					print (len(y_pred))
					print (y_pred)
					print (cnt)
					# RES[:,cnt] = y_pred
					# print ("RES", np.array(RES).shape)
					RES[label] = y_pred
					cnt+=1 
					print(RES)
				i += 1


		# feature_importances_dict = {}
		feature_importances_df = pd.DataFrame()

		for (est_name, est) in selected_estimators_vitality:
			print ("************************")
			print ("cross-validating vitality from JJ vars with ", est_name )
			print ("************************")
			Xv = RES.values
			yv = new_y
			score = cross_validate(est, Xv, yv,
				scoring=['r2', 'neg_mean_absolute_error'],
				n_jobs=-1, verbose=0, cv=5,
				return_estimator =True)

			print(np.mean(score['test_r2']),
				np.std(score['test_r2']),
				-np.mean(score['test_neg_mean_absolute_error']),
				np.std(score['test_neg_mean_absolute_error']))

			for idx,estimator in enumerate(score['estimator']):
				print("Features sorted by their score for estimator {}:".format(idx))
				feature_importances = pd.DataFrame(estimator.feature_importances_,
												   # index = land_use_cols + small_blocks_cols,
												   index = RES.columns,
													columns=['importance']).sort_values('importance', ascending=False)
				# print(feature_importances)

				if idx == 0:
					feature_importances_df = feature_importances
				else:
					feature_importances_df = feature_importances_df + feature_importances



			print (feature_importances_df.sort_values('importance', ascending=False))


		if rnd_st == 0:
			FI_df = feature_importances_df
		else:
			FI_df = feature_importances_df + FI_df


	print ("Final Feature Importances ")
	print (FI_df.sort_values('importance', ascending=False))

	FI_df = FI_df / 50.0

	print (FI_df.sort_values('importance', ascending=False))
		# print (cnt)
		# print (RES)
		# print(y)


def plot_different_leave_out_cities(data=data, city='all', label="label_activity_density"):

	# params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
 #          'learning_rate': 0.01, 'loss': 'ls'}

	# estimators = [
	# ('LinearRegression', LinearRegression()),
 #    ('ElasticNet', ElasticNetCV()),
 #    ('SVM regressor', SVR()),
 #    ('Ridge', RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])),
 #    ('Random Forest', RandomForestRegressor(random_state=42)),
 #    ('XGBoost', GradientBoostingRegressor(**params))
	# ]	

	# est = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
	est = SVR()
	cities = ['milano', 'bologna', 'roma', 'torino', 'firenze', 'palermo']


	fig, axs = plt.subplots(2, 3, figsize=(14, 10))
	axs = np.ravel(axs)

	for ax, city in zip(axs, cities):
		start_time = time.time()

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

		est.fit(X, y)	
		y_pred = est.predict(X_city)

		elapsed_time = time.time() - start_time

		r2 = r2_score(y_city, y_pred)
		mae = mean_absolute_error(y_city, y_pred)
		
		plot_regression_results(
			ax, y_city, y_pred,
			city,
			(r'$R^2={:.2f} $' + '\n' + r'$MAE={:.2f} $')
			.format(r2,
					mae),
			elapsed_time)

	# plt.suptitle('Single predictors versus stacked predictors')
	plt.tight_layout()
	plt.subplots_adjust(top=0.9)
	plt.savefig("results/prediction/districts/regression/different_cities.png", dpi=100)
	plt.show()



# for l in label_columns:
# 	plot_different_estimators(label=l)

# plot_different_leave_out_cities(data=data, label="label_activity_density")

# regress_variables()

# plot_different_estimators(label="label_activity_density")

regress_vitality_from_variables()
