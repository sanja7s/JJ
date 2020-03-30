import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import json


data_dir = "preprocessed/" 
features_dir = data_dir + "features/milano/"
features_file = "VGG16/df_VGG16_feat8192_pca"
labels_dir = data_dir + "labels/milano/" 

PCA_components = 32

label_columns = ["hType_mix", "num_intersect", "bld_avg_age", "emp_rat_num",\
				"LUM5_single",	"RNR_nres", "mdist_smallparks", "nig_rat_daily",\
				"nig_rat_daily3", "mdist_nres_daily", "num_community_places", \
				"num_community_places_poi", "avg_block_area", "sphi", \
				"enterprises_empl_size", "pop_rat_num",  \
				"emp_rat_pop", "bld_rat_area", "den_nres_daily",\
				"mdist_parks", "den_nres_non-daily", "mdist_railways",\
				"mdist_highways", "mdist_water", "activity_density"]



def clean_labels_features():
	
	data = pd.read_csv(labels_dir + "imagelet_labels.csv")
	data2 = data[data != -1]
	data3 = data2.dropna()
	data3 = data3.rename(columns={"imagelet":"imageName"})
	data3["imageName"] = data3["imageName"].apply(lambda x: x.replace(".shp", ""))
	data3 = data3.drop(columns= [ c for c in data3.columns if "Unnamed" in c])
	data3.to_csv(labels_dir + "imagelet_labels_clean.csv", index=0)
	
	data1 = pd.read_csv(features_dir + features_file\
		 + str(PCA_components) + ".csv")
	data1 = data1.rename(columns = {"name":"imageName"})
	data1["imageName"] = data1["imageName"].apply(lambda x: x.replace(".jpg", ""))
	data4 = pd.merge(data1,data3, on="imageName", how="inner")
	for l in label_columns:
		try:
			data4 =  data4.drop(columns=["label_" + l])
		except:
			print ("could not drop column ", l)
			pass
	data4 = data4.drop(columns= [ c for c in data4.columns if "Unnamed" in c])
	data4.to_csv(features_dir + features_file+\
			str(PCA_components) + "_clean.csv", index=0)



def create_normalized_labels_features():

	clean_labels_features()
	
	labels = pd.read_csv(labels_dir + "imagelet_labels_clean.csv")
	features = pd.read_csv(features_dir + features_file+\
		str(PCA_components) + "_clean.csv")
	# print (features)
	features2 = features.set_index("imageName")
	standardized_features = StandardScaler().fit_transform(features2)
	standardized_features_df = \
			pd.DataFrame(standardized_features,columns=\
				['PCA%i' % i for i in range(len(features2.columns))])
	features = pd.concat([features["imageName"], standardized_features_df], axis = 1)
	data = pd.merge(features,labels, on="imageName", how="inner")

	return data.reset_index()



d1 = create_normalized_labels_features()
d1.to_csv(features_dir + features_file+\
		str(PCA_components) +"_labels_features.csv", index=0)


