import pandas as pd



# network_type = "vgg16_4096"

# PCA_components = 64

# LABELING_METHOD = "maximum"
# AVERAGING_METHOD = "kaist"

# normalize = False


norm_dict = {
	True:"normalized_",
	False: "",
	"Total": "normalized_across_all_"
}



data_dir = "preprocessed/" 




def get_training_data(network_type, PCA_components,\
				LABELING_METHOD, AVERAGING_METHOD,\
				pred_type, normalize, data_dir=data_dir):

	f = get_features(network_type, PCA_components,\
				LABELING_METHOD, AVERAGING_METHOD, data_dir)

	l = get_labels(pred_type, normalize, data_dir)

	df = f.merge(l, on=["city_district"])

	# print (df.head())
	return df


def get_features(network_type, PCA_components,\
				LABELING_METHOD, AVERAGING_METHOD, data_dir):

	features_dir = data_dir + "training_data/features/districts/"

	if network_type == "vgg19":
		features_file = "Italy_6_cities_vgg19_pca"+str(PCA_components)+"_linear_fc_thirdlast_layer.csv"
	elif network_type == "resnet50":
		features_file = "Italy_6_cities_resnet_pca"+str(PCA_components)+"_second_last_layer.csv"
	elif network_type == "vgg16_4096":
		features_file = "Italy_6_cities_resnet_pca" + str(PCA_components) + "_vgg16_4096.csv"


	ff  = features_file.replace(".csv", "_" + \
		LABELING_METHOD + "_" + AVERAGING_METHOD +"_features.csv")
	features = pd.read_csv(features_dir + ff)

	# print (features.head())
	return features


def get_labels(pred_type, normalize, data_dir):

	label_dir = data_dir + "training_data/labels/districts/"
	
	ll = norm_dict[normalize] + "district_" \
		+ pred_type + "_labels.csv"
	labels = pd.read_csv(label_dir + ll)

	labels.rename(columns={l:"label_" + l for l in \
		labels.columns if l != "city_district"},\
		inplace=True)

	# print (labels.head())
	return labels

# get_training_data()