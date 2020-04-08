#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sanja7s


"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

n_components = 8
# n_features = 2048

network_type = "vgg16_4096"

data_dir = "preprocessed/" 
features_dir = data_dir + "features/features_all/"
if network_type == "vgg19":
	features_file = "Italy_6_cities_vgg19_linear_fc_thirdlast_layer.json"
elif network_type == "resnet50":
	features_file = "Italy_6_cities_resnet50_layer4_flattened.json"
elif network_type == "vgg16_4096":
	features_file = "Italy_6_cities_vgg16_4096_flattened.json"
elif network_type == "vgg16_2048":
	features_file = "Italy_6_cities_vgg16_2048_flattened.json"
elif network_type == "vgg16_2048_v1": # from stephen
	features_file = "df_VGG16Featuress_2048_all1.csv"
elif network_type == "vgg16_2048_v2":  # from stephen
	features_file = "df_VGG16Featuress_2048_all2.csv"
elif network_type == "vgg16_8192_v1": # from stephen
	features_file = "df_VGG16Featuress_8192_all1.csv"
elif network_type == "vgg16_8192_v2":  # from stephen
	features_file = "df_VGG16Featuress_8192_all2.csv"

def read_features():
	if network_type in ["vgg16_2048_v1", "vgg16_2048_v2", "vgg16_8192_v1", "vgg16_8192_v2"]:
		data = pd.read_csv(features_dir + features_file)
	else:
		data = pd.read_json(features_dir + features_file)
		print (len(data.loc[0, 'feature']))
	return data



resnet_features = read_features()

if network_type in ["vgg16_2048_v1", "vgg16_2048_v2", "vgg16_8192_v1", "vgg16_8192_v2"]:
	n_features = len(resnet_features.columns)-1
else:
	n_features = len(resnet_features.loc[0, 'feature'])

print (n_features)

# # uncomment to test for Milano city only
# ##########################################
# print ("BEFORE ", len(resnet_features))
# resnet_features = resnet_features[resnet_features["imageName"].str.contains("S2B_MSIL2A_20181024T102059_N0209_R065_T32TNR_20181024T171522")]
# print ("AFTER ", len(resnet_features))
# ##########################################

resnet_features = resnet_features.set_index("imageName")

rf = resnet_features.copy()

if network_type in ["vgg16_2048_v1", "vgg16_2048_v2", "vgg16_8192_v1", "vgg16_8192_v2"]:
	rf = rf.rename(columns={'f'+str(i):i for i in rf.columns if i != "imageName"})
else:
	rf[['f'+str(i) for i in range(n_features)]] = \
		pd.DataFrame(rf.feature.values.tolist(), \
		index= rf.index)

	rf = rf.drop(columns=['feature'])

print (rf.head())

standardized_features = StandardScaler().fit_transform(rf)
print (standardized_features)

standardized_features_df = pd.DataFrame(standardized_features)

rf=rf.reset_index()

resnet_features_standardized = \
	pd.concat([ rf["imageName"], standardized_features_df], axis = 1)

print (resnet_features_standardized)

resnet_features_standardized = resnet_features_standardized.set_index("imageName")

pca = PCA(n_components=n_components)
pca.fit(resnet_features_standardized)

pca_df = pd.DataFrame(pca.transform(resnet_features_standardized), \
	columns=['PCA%i' % i for i in range(n_components)],\
	index=resnet_features_standardized.index)

print (pca.explained_variance_ratio_)
print (sum(pca.explained_variance_ratio_), len(pca.explained_variance_ratio_))

pca_df = pca_df.reset_index()


if network_type == "vgg19":
	pca_df.to_csv(features_dir+"Italy_6_cities_vgg19" + \
		"_pca" + str(n_components) + "_linear_fc_thirdlast_layer.csv")
elif network_type == "resnet50":
	pca_df.to_csv(features_dir+"Italy_6_cities_resnet" + \
		"_pca" + str(n_components) + "_second_last_layer.csv")
elif network_type == "vgg16_4096":
	pca_df.to_csv(features_dir+"Italy_6_cities_resnet" + \
		"_pca" + str(n_components) + "_vgg16_4096.csv")
elif network_type == "vgg16_2048":
	pca_df.to_csv(features_dir+"Italy_6_cities_resnet" + \
		"_pca" + str(n_components) + "_vgg16_2048.csv")

