import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

CITY = "milano"
data_dir = "preprocessed/" 
features_dir = data_dir + "features/" + CITY + "/"
labels_dir = data_dir + "labels/" + CITY + "/"

PCA_components = 32

label_columns = ["hType_mix", "num_intersect", "bld_avg_age", "emp_rat_num",\
				"LUM5_single",	"RNR_nres", "mdist_smallparks", "nig_rat_daily",\
				"nig_rat_daily3", "mdist_nres_daily", "num_community_places", \
				"num_community_places_poi", "avg_block_area", "sphi", \
				"enterprises_empl_size", "pop_rat_num",  \
				"emp_rat_pop", "bld_rat_area", "den_nres_daily",\
				"mdist_parks", "den_nres_non-daily", "mdist_railways",\
				"mdist_highways", "mdist_water"]



def clean_labels_features():
    
    data = pd.read_csv(labels_dir + "imagelet_labels.csv")
    data2 = data[data != -1]
    data3 = data2.dropna()
    data3 = data3.rename(columns={"imagelet":"name"})
    data3["name"] = data3["name"].apply(lambda x: x.replace(".shp", ""))
    data3 = data3.drop(columns= [ c for c in data3.columns if "Unnamed" in c])
    data3.to_csv(labels_dir + "imagelet_labels_clean.csv", index=0)
    
    data1 = pd.read_csv(features_dir + "Resnet50/df_ResNet50_feat8192_pca"\
    	 + str(PCA_components) + ".csv")
    data1["name"] = data1["name"].apply(lambda x: x.replace(".jpg", ""))
    data4 = pd.merge(data1,data3, on="name", how="inner")
    data4 =  data4.drop(columns=["label_" + l for l in label_columns])
    data4 = data4.drop(columns= [ c for c in data4.columns if "Unnamed" in c])
    data4.to_csv(features_dir + "Resnet50/df_ResNet50_feat8192_pca"+\
    		str(PCA_components) +".csv", index=0)



def create_normalized_labels_features():
    
    labels = pd.read_csv(labels_dir + "imagelet_labels_clean.csv")
    features = pd.read_csv(features_dir + "Resnet50/df_ResNet50_feat8192_pca"+\
    	str(PCA_components) + ".csv")
    features2 = features.set_index("name")
    standardized_features = StandardScaler().fit_transform(features2)
    standardized_features_df = \
    		pd.DataFrame(standardized_features,columns=\
                ['PCA%i' % i for i in range(len(features2.columns))])
    features = pd.concat([features["name"], standardized_features_df], axis = 1)
    data = pd.merge(features,labels, on="name", how="inner")
    data.to_csv(features_dir + "Resnet50/df_ResNet50_feat8192_pca"+\
    		str(PCA_components) +"_labels_features.csv", index=0)



clean_labels_features()
create_normalized_labels_features()