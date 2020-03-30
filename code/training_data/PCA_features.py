#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sanja7s


"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

n_components = 16
CITY = "milano"

def PCA_resnet():
	features_dir = "preprocessed/features/" + CITY + "/Resnet50/"

	resnet_features = \
		pd.read_csv(features_dir + "df_ResNet50_feat8192.csv")

	resnet_features = resnet_features.set_index("name")

	print (resnet_features.head())

	standardized_features = StandardScaler().fit_transform(resnet_features)

	print (standardized_features)

	standardized_features_df = pd.DataFrame(standardized_features)

	resnet_features=resnet_features.reset_index()

	resnet_features_standardized = \
		pd.concat([ resnet_features["name"], standardized_features_df], axis = 1)

	print (resnet_features_standardized)

	resnet_features_standardized = resnet_features_standardized.set_index("name")

	pca = PCA(n_components=n_components)
	pca.fit(resnet_features_standardized)

	pca_df = pd.DataFrame(pca.transform(resnet_features_standardized), \
		columns=['PCA%i' % i for i in range(n_components)],\
		index=resnet_features_standardized.index)

	print (pca.explained_variance_ratio_)

	print (sm(pca.explained_variance_ratio_), len(pca.explained_variance_ratio_))


	pca_df = pca_df.reset_index()

	pca_df.to_csv(features_dir+"df_ResNet50_feat8192" + \
		"_pca" + str(n_components) + ".csv")



def PCA_VGG():
	features_dir = "preprocessed/features/" + CITY + "/VGG16/"

	vgg_features = \
		pd.read_csv(features_dir + "df_VGG16_feat8192.csv")

	vgg_features = vgg_features.set_index("name")

	print (vgg_features.head())

	standardized_features = StandardScaler().fit_transform(vgg_features)

	print (standardized_features)

	standardized_features_df = pd.DataFrame(standardized_features)

	vgg_features=vgg_features.reset_index()

	vgg_features_standardized = \
		pd.concat([ vgg_features["name"], standardized_features_df], axis = 1)

	print (vgg_features_standardized)

	vgg_features_standardized = vgg_features_standardized.set_index("name")

	pca = PCA(n_components=n_components)
	pca.fit(vgg_features_standardized)

	pca_df = pd.DataFrame(pca.transform(vgg_features_standardized), \
		columns=['PCA%i' % i for i in range(n_components)],\
		index=vgg_features_standardized.index)

	print (pca.explained_variance_ratio_)

	print (sum(pca.explained_variance_ratio_), len(pca.explained_variance_ratio_))


	pca_df = pca_df.reset_index()

	pca_df.to_csv(features_dir+"df_VGG16_feat8192" + \
		"_pca" + str(n_components) + ".csv")


PCA_VGG()
