import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pearsonr
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

data_dir = "preprocessed/" 
features_in_dir = data_dir + "features/features_all/"
features_out_dir = data_dir + "regression_features/features_all/"
labels_dir = data_dir + "district_labels/"
separate_features_dir = data_dir + "regression_features/features_separate_cities/"

output_dir = data_dir + "district_features/regression/"

network_type = "vgg16_4096" 

PCA_components = 8

LABELING_METHOD = "threshold"
AVERAGING_METHOD = "average"

features_columns = ["PCA"+str(i) for i in range(PCA_components)]
geo_columns = ["centroid_x", "centroid_y"]

USE_GEO = "ONLY_GEO"
if USE_GEO == "ALSO_GEO" and AVERAGING_METHOD != "kaist":
	features_columns = features_columns + ["centroid_x", "centroid_y"]
elif USE_GEO == "ONLY_GEO":
	features_columns = geo_columns


label_columns = ["hType_mix", "num_intersect", "bld_avg_age", "emp_rat_num",\
				"LUM5_single",	"RNR_nres", "mdist_smallparks", "nig_rat_daily",\
				"nig_rat_daily3", "mdist_nres_daily", "num_community_places", \
				"num_community_places_poi", "avg_block_area", "sphi", \
				"enterprises_empl_size", "pop_rat_num",  \
				"emp_rat_pop", "den_nres_daily",\
				"mdist_parks", "den_nres_non-daily", "mdist_railways",\
				"mdist_highways", "mdist_water", "activity_density"]
label_columns = ["label_" + l for l in label_columns]

if network_type == "vgg19":
	features_file = "Italy_6_cities_vgg19_pca"+str(PCA_components)+"_linear_fc_thirdlast_layer.csv"
elif network_type == "resnet50":
	features_file = "Italy_6_cities_resnet_pca"+str(PCA_components)+"_second_last_layer.csv"
elif network_type == "vgg16_4096":
	features_file = "Italy_6_cities_resnet_pca" + str(PCA_components) + "_vgg16_4096.csv"
geo_features_file = "all_imagelets_geocoord.csv"



def create_normalized_labels_features(city_name, method=AVERAGING_METHOD):

	df = pd.read_csv(separate_features_dir + \
	features_file.replace(".csv", "_" + city_name \
		+ "_labels_features.csv"))

	labels = pd.read_csv(labels_dir + city_name +\
		 "_imagelet_labels_" + LABELING_METHOD + ".csv")

	print (sum(labels["label_district"]))

	labels = labels.rename(columns={"imagelet":"imageName"})
	labels["imageName"] = labels["imageName"].\
		apply(lambda x: x.replace(".shp",""))

	merged_df = df.merge(labels, on ='imageName', how="inner")

	if method == "average":
		district_df = merged_df[label_columns+features_columns+["label_district"]]\
			.groupby(["label_district"]).mean()
		district_df = district_df.reset_index()
		# del district_df["Unnamed: 0"]
		# del district_df["index"]
		saved_district_df = district_df.drop(columns=features_columns)

		standardized_features = StandardScaler().fit_transform(district_df[features_columns])
		minmax_features = MinMaxScaler().fit_transform(standardized_features)
		standardized_minmax_features_df = \
				pd.DataFrame(minmax_features,columns=features_columns)
		fl = pd.concat([saved_district_df, standardized_minmax_features_df], axis = 1)
	elif method == "kaist": # paper Lightweight and Robust Representation of Economic Scales from Satellite Imagery
		# labels should stay as they are
		label_df = merged_df[label_columns+["label_district"]].\
			groupby(["label_district"]).mean()
		# features will have 2 new aggregation values: first is mean
		features_df_avg = merged_df[features_columns+["label_district"]].\
			groupby("label_district").agg(np.mean)
		# second is std
		features_df_std = merged_df[features_columns+["label_district"]].\
			groupby("label_district").agg(lambda x: np.nanstd(x, ddof=0))
		# give the names to these new features
		features_df_avg.rename\
			(columns={l:"f_avg_"+l for l in features_df_avg.columns}, inplace=True)
		features_df_std.rename\
			(columns={l:"f_std_"+l for l in features_df_std.columns}, inplace=True)
		# add one column which is only count
		count = merged_df[["label_district"]].\
			groupby("label_district").size().reset_index(name="f_count")
		count = count.set_index("label_district")

		features_df = features_df_std.join(features_df_avg)
		features_df = features_df.join(count)
		if USE_GEO == "ALSO_GEO":
			# geo should be merged
			geo_df = merged_df[geo_columns+["label_district"]].\
				groupby(["label_district"]).mean()
			features_df = features_df.join(geo_df)
			features_df.rename\
				(columns={l:"f_"+l for l in geo_df.columns}, inplace=True)
			
		district_df = features_df.join(label_df)
	
		fl = district_df.reset_index()


	fl["city"] = city_name
	# del district_df["index"]
	return fl



cities = ['palermo', 'torino', 'bologna','firenze', 'roma']
city_name="milano"
d1 = create_normalized_labels_features(city_name)
print (d1)
d1.to_csv(output_dir + \
	features_file.replace(".csv", "_" + LABELING_METHOD +\
		"_" + AVERAGING_METHOD + \
		"_" + city_name + "_labels_features.csv"), index=0)
for city_name in cities:
	print (city_name)
	d2 = create_normalized_labels_features(city_name)
	# print (d2)
	d2.to_csv(output_dir + \
		features_file.replace(".csv", "_" + LABELING_METHOD +\
		"_" + AVERAGING_METHOD + \
		"_" + city_name + "_labels_features.csv"), index=0)
	d1 = pd.concat([d1, d2], ignore_index=True, sort=False)
	print ('done')
d1.to_csv(output_dir + \
	features_file.replace(".csv", "_" + \
		LABELING_METHOD + "_" + AVERAGING_METHOD +"_labels_features.csv"), index=0)


