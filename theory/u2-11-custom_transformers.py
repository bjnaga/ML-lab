# Custom Transformers
# create a class and implement fit(), transform(), fit_transform() methods
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
 

file_path = os.path.join("datasets","housing","housing.csv")


housing = pd.read_csv(file_path)

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator ,TransformerMixin):
    def __init__(self,add_bedtooms_per_room = True):
        self.add_bedtooms_per_room = add_bedtooms_per_room
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        rooms_per_household = X[:,rooms_ix]/X[:,households_ix]
        population_per_household = X[:,population_ix]/X[:,households_ix]
        if self.add_bedtooms_per_room:
            bedrooms_per_room = X[:,bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
        else:
            return np.c_[X,rooms_per_household,population_per_household]
attr_adder = CombinedAttributesAdder(add_bedtooms_per_room= True)
housing_extra_attribs = attr_adder.transform(housing.values)
print(housing_extra_attribs.shape)
print(housing_extra_attribs[0])
print(housing.head())

print(housing.columns)
new_housing=housing.drop(['ocean_proximity'], axis=1)





# Normalization X scaled = (Xi - Xmean)/(Xmax - Xmin)
# Min-Max Scaling X scale = (Xi - Xmin)/(Xmax - Xmin)
# Absolute Maximum Scaling  = (Xi - max(|X|)/max(|X|)

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
