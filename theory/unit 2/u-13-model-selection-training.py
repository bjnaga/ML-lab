from sklearn.linear_model import LinearRegression
import os 
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np



file_path = os.path.join("datasets","housing","housing.csv")

housing = pd.read_csv(file_path)
median = housing["total_bedrooms"].median()
print(median)
housing["total_bedrooms"].fillna(median,inplace=True)
housing_num = housing.drop("ocean_proximity", axis=1)
# strat_train_set=pd.DataFrame()

# strat_test_set = pd.DataFrame()


housing_num['income_cat'] = pd.cut(x=housing_num['median_income'], bins=[0, 1.5, 3, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5])
print(housing_num.head())
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(X=housing_num, y=housing_num['income_cat']):
    strat_train_set = housing_num.loc[train_index]
    strat_test_set = housing_num.loc[test_index]

print(strat_test_set.head())
# for set_ in (strat_train_set, strat_test_set):
#     set_.drop('income_cat', axis=1, inplace=True)

print(strat_train_set.head())
strat_train_set_labels = strat_train_set["median_house_value"].copy()
strat_train_set = strat_train_set.drop("median_house_value", axis=1)

strat_test_set_labels = strat_test_set["median_house_value"].copy()
strat_test_set = strat_test_set.drop("median_house_value", axis=1)

print("#########################################")
print(strat_train_set)

lin_reg = LinearRegression()
lin_reg.fit(X=strat_train_set, y=strat_train_set_labels)

# trying on the model on training set 
some_data = strat_train_set.iloc[:5]
some_labels = strat_train_set_labels.iloc[:5]

print("Predictions : ",lin_reg.predict(some_data))

print("Labels :", list(some_labels))



# RMSE 
from sklearn.metrics import mean_squared_error

housing_prediction  = lin_reg.predict(strat_train_set)

lin_mse = mean_squared_error(strat_train_set_labels,housing_prediction)
lin_rmse = np.sqrt(lin_mse)
print("RMSE for linear regression",lin_rmse)



# DecisionTreeRegression

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(strat_train_set,strat_train_set_labels)
housing_prediction_DTR = tree_reg.predict(strat_train_set)
tree_mse = mean_squared_error(strat_train_set_labels,housing_prediction_DTR)
tree_rmse = np.sqrt(tree_mse)
print("RMSE for train set  decision tree regression ",tree_rmse)

# housing_prediction_DTR = tree_reg.predict(strat_test_set)
# tree_mse = mean_squared_error(strat_test_set_labels,housing_prediction_DTR)
# tree_rmse = np.sqrt(tree_mse)
# print("RMSE for test set decision tree regression ",tree_rmse)

# Better Evaluation using Cross_Validation using K fold cross validation feature

from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg,strat_train_set,strat_train_set_labels,scoring="neg_mean_squared_error",cv =10)
tree_rmse_scores = np.sqrt(-scores)
print("\n\ntree rms score 10 validations are : \n",tree_rmse_scores)
print("\n\nmean : \n",tree_rmse_scores.mean())
print("\n\nstd deviation : \n",tree_rmse_scores.std())



scores = cross_val_score(lin_reg,strat_train_set,strat_train_set_labels,scoring="neg_mean_squared_error",cv =10)
lin_rmse_scores = np.sqrt(-scores)
print("\n\ntree rms score 10 validations are : \n",lin_rmse_scores)
print("\n\nmean : \n",lin_rmse_scores.mean())
print("\n\nstd deviation : \n",lin_rmse_scores.std())


# ensemble learning using RandomForestRegressor 
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_model = forest_reg.fit(strat_train_set,strat_train_set_labels)
forest_prediction =forest_model.predict(strat_train_set)
forest_mse = mean_squared_error(strat_train_set_labels,forest_prediction)
forest_rmse = np.sqrt(forest_mse)
print("RMSE for train set  Random forest regressor ",forest_rmse)


scores = cross_val_score(forest_reg,strat_train_set,strat_train_set_labels,scoring="neg_mean_squared_error",cv =10)
forest_rmse_scores = np.sqrt(-scores)
print("\n\nRandom forest regressor score 10 validations are : \n",forest_rmse_scores)
print("\n\nmean : \n",forest_rmse_scores.mean())
print("\n\nstd deviation : \n",forest_rmse_scores.std())
