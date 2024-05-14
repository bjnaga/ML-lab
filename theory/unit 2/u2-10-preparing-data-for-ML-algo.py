# x=range(2,10,2)
# for x in range(35,84,2):
#     print(str(x)+",",end="")
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer

file_path = os.path.join("datasets","housing","housing.csv")


housing = pd.read_csv(file_path)
housing["median_income"].hist()
housing["income_cat"] = pd.cut(housing["median_income"],bins=[0.,1.5,3.,4.5,6,np.inf],labels=[1,2,3,4,5])
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=21)
for train_index, test_index in split.split(housing,housing["income_cat"]): #Generate indices to split data into training and test set.
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
print(strat_train_set)
print(strat_test_set)

# print(strat_test_set["income_cat"].value_counts())

# print(strat_test_set["income_cat"].value_counts()/len(strat_test_set))


# lets drop the income_cat column as we have to stratified sampling data (representing the actual data set)
 
for set_ in (strat_test_set,strat_train_set):
    set_.drop("income_cat",axis=1,inplace=True)
print(strat_test_set)
print(strat_train_set)

# Data Cleaning

housing = strat_train_set.drop("median_house_value",axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# missing values 
    # get rid of corresponding districts
    # get rid of whole attribute 
    # set value to some value (0 or mean or median )

# drop(), dropna(), fillna()
print("#######################################################################################################")
print(housing.dropna(subset=["total_bedrooms"]))
print(housing.drop("total_bedrooms",axis=1))
median = housing["total_bedrooms"].median()
print(median)
housing["total_bedrooms"].fillna(median,inplace=True)
print(housing["total_bedrooms"])
print(housing.info())
# df.method({col: value}, inplace=True)
# print(housing.method({total_bedrooms:median} ,inplace=True))
# df[col] = df[col].method(value)
# housing["total_bedrooms"]=housing["total_bedrooms"].method(median)
# print(housing["total_bedrooms"].method(median))
# print(housing.info())

# Simple Imputer Class


imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity",axis=1)
print(imputer.fit(housing_num))
print(imputer.statistics_)
print(housing_num.median().values)

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X,columns=housing_num.columns,index=housing_num.index)
print(housing_tr.info())
print(housing_tr)


# Handling text and Categorical Attributes
housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(10))


from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:10])
print("Categories of the encoders are:",ordinal_encoder.categories_)

# converting to 0 or 1 

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1Hot = cat_encoder.fit_transform(housing_cat)
# print(housing_cat_1Hot)
print(housing_cat_1Hot.toarray())
print(cat_encoder.categories_)





