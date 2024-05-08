import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

file_path = os.path.join("datasets","housing","housing.csv")

housing = pd.read_csv(file_path)

print("#######################################################################################################")
print("\n\n---Dropping null values---\n")
print(housing.dropna(subset=["total_bedrooms"]).describe())

print("\n\n---Dropping the entire column---\n")
print(housing.drop("total_bedrooms",axis=1))
median = housing["total_bedrooms"].median()
print(median)

print("\n\n---Filling null values with median value---\n")
housing["total_bedrooms"].fillna(median,inplace=True)
print(housing["total_bedrooms"])

print("\n\n---Printing Dataset information---\n")
print(housing.info())

# Simple Imputer Class

imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity",axis=1)

imputer.fit(housing_num)
print("\n\n---Median values for numerical columns using statistics---\n")
print(imputer.statistics_)
print("#########################################################")
print("\n\n---Median values for numerical columns---\n")
print(housing_num.median().values)

X = imputer.transform(housing_num)
print("#########################################################")
housing_tr = pd.DataFrame(X,columns=housing_num.columns,
                          index=housing_num.index)
print("\n\n---Tranformed data---\n")
print(housing_tr.info())
print(housing_tr)

print("\n\n Handling text and Categorical Attributes\n")
housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(10))

print("\n\n---Ordinal Encoding---\n")
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:10])
print("Categories of the encoders are:",ordinal_encoder.categories_)

# converting to 0 or 1 

print("\n\n---One Hot Encoding---\n")
cat_encoder = OneHotEncoder()
housing_cat_1Hot = cat_encoder.fit_transform(housing_cat)
# print(housing_cat_1Hot)
print(housing_cat_1Hot.toarray())
print(cat_encoder.categories_)

print("\n\n---Feature Scaling---\n")
print("########################################")
new_housing=housing.drop(['ocean_proximity'], axis=1)
print(new_housing.head())
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pd.DataFrame(new_housing))

scaled_df = pd.DataFrame(scaled_data,
                         columns=['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'median_house_value'])
print(scaled_df.head())
print(scaled_data)




