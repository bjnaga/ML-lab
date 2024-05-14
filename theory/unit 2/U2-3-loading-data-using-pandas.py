import pandas as pd
import os
import matplotlib.pyplot as plt
file_path = os.path.join("datasets","housing")
os.makedirs(file_path,exist_ok=True)
file_path = os.path.join(file_path,"housing.csv")

# this function returns data frame object 
def load_housing_data(housing_path = file_path):
    return pd.read_csv(housing_path)

# print(load_housing_data(file_path))
# print(load_housing_data(file_path).head())
# print(load_housing_data(file_path).info())
# print()

housing = load_housing_data(file_path)
print(housing.head())
print(housing.info())
print(housing["ocean_proximity"].value_counts())
print(housing.describe())

housing.hist()
housing.hist(bins=50)
housing.hist(bins=50,figsize=(20,15))
plt.show()
