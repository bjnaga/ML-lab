from sklearn.linear_model import LinearRegression
import os 
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from sklearn.metrics import mean_squared_error

file_path = os.path.join("datasets","housing","housing.csv")
housing = pd.read_csv(file_path)
median = housing["total_bedrooms"].median()
print(median)
housing["total_bedrooms"].fillna(median,inplace=True)
housing_num = housing.drop("ocean_proximity", axis=1)

housing_num['income_cat'] = pd.cut(x=housing_num['median_income'], bins=[0, 1.5, 3, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5])
print(housing_num.head())
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(X=housing_num, y=housing_num['income_cat']):
    strat_train_set = housing_num.loc[train_index]
    strat_test_set = housing_num.loc[test_index]

print(strat_test_set.head())

print(strat_train_set.head())
strat_train_set_labels = strat_train_set["median_house_value"].copy()
strat_train_set = strat_train_set.drop("median_house_value", axis=1)

strat_test_set_labels = strat_test_set["median_house_value"].copy()
strat_test_set = strat_test_set.drop("median_house_value", axis=1)

print("#########################################")
lin_reg = LinearRegression()
lin_reg.fit(X=strat_train_set, y=strat_train_set_labels)
housing_prediction  = lin_reg.predict(strat_train_set)

lin_mse = mean_squared_error(strat_train_set_labels,housing_prediction)
lin_rmse = np.sqrt(lin_mse)
print("RMSE for linear regression",lin_rmse)