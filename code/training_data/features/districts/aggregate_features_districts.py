import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pearsonr
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

"""
	from features of imagelets, 
	this module creates 4 possible
	aggregated features of districts:
	1) threshold average
	2) threshold kaist
	3) maximum average
	4) maximum kaist
	geo features are simply appended
	all features are normalized using
	standard scaling and minmax 
"""

network_type = "vgg16_4096" 

PCA_components = 10

LABELING_METHOD = "maximum"
AVERAGING_METHOD = "kaist"


def aggregate_all(PCA_components):

	if network_type == "vgg19":
		features_file = "Italy_6_cities_vgg19_pca"+str(PCA_components)+"_linear_fc_thirdlast_layer.csv"
	elif network_type == "resnet50":
		features_file = "Italy_6_cities_resnet_pca"+str(PCA_components)+"_second_last_layer.csv"
	elif network_type == "vgg16_4096":
		features_file = "Italy_6_cities_resnet_pca" + str(PCA_components) + "_vgg16_4096.csv"
	geo_features_file = "all_districts_centroids.csv"


	def create_normalized_features(method=AVERAGING_METHOD):

		df = pd.read_csv(features_in_dir + features_file)
		df["imageName"] = df["imageName"].\
			apply(lambda x: x.replace(".jpg",""))

		geo_df = pd.read_csv(features_out_dir + geo_features_file)
		geo_df = geo_df.set_index("city_district")

		labels = pd.read_csv(labels_dir + \
			"all_imagelet_labels_" + LABELING_METHOD + ".csv")
		del labels["label_district"]

		labels = labels.rename(columns={"imagelet":"imageName"})
		labels["imageName"] = labels["imageName"].\
			apply(lambda x: x.replace(".shp",""))

		print (df.columns)
		print (labels.columns)

		merged_df = df.merge(labels, on ='imageName', how="inner")

		print (merged_df.head())

		if method == "average":
			features_df = merged_df.groupby(["city_district"]).mean()

		elif method == "kaist": # paper Lightweight and Robust Representation of Economic Scales from Satellite Imagery
			# features will have 2 new aggregation values: first is mean
			features_df_avg = merged_df[features_columns+["city_district"]].\
				groupby("city_district").agg(np.mean)
			# second is std
			features_df_std = merged_df[features_columns+["city_district"]].\
				groupby("city_district").agg(lambda x: np.nanstd(x, ddof=0))
			# give the names to these new features
			features_df_avg.rename\
				(columns={l:"f_avg_"+l for l in features_df_avg.columns}, inplace=True)
			features_df_std.rename\
				(columns={l:"f_std_"+l for l in features_df_std.columns}, inplace=True)
			# add one column which is only count
			count = merged_df[["city_district"]].\
				groupby("city_district").size().reset_index(name="f_count")
			count = count.set_index("city_district")

			features_df = features_df_std.join(features_df_avg)
			features_df = features_df.join(count)


		features_df = features_df.join(geo_df, how="inner")
		features_df = features_df.reset_index()

		all_features_columns = [c for c in features_df.columns if c != "city_district"]

		saved_district_df = features_df.drop(columns=all_features_columns)

		standardized_features = StandardScaler().fit_transform(features_df[all_features_columns])
		minmax_features = MinMaxScaler().fit_transform(standardized_features)
		standardized_minmax_features_df = \
				pd.DataFrame(minmax_features,columns=all_features_columns)
		fl = pd.concat([saved_district_df, standardized_minmax_features_df], axis = 1)

		fl.rename\
			(columns={l:"f_"+l for l in geo_columns}, inplace=True)
				
		return fl


	data_dir = "preprocessed/" 
	features_in_dir = data_dir + "training_data/features/imagelets/"
	features_out_dir = data_dir + "training_data/features/districts/"
	# these are only the imagelet labels
	labels_dir = data_dir + "training_data/district_imagelets_link/"


	d1 = create_normalized_features()
	d1.to_csv(features_out_dir + \
		features_file.replace(".csv", "_" + \
		LABELING_METHOD + "_" + AVERAGING_METHOD +"_features.csv"), index=0)

for PCA_components in [8,9,10,11,12,13,14,15,16,17,18,19,20,26,32,48,64]:
	features_columns = ["PCA"+str(i) for i in range(PCA_components)]
	geo_columns = ["centroid_x", "centroid_y"]
	aggregate_all(PCA_components)
