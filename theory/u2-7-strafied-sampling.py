import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
file_path = os.path.join("datasets","housing","housing.csv")

housing = pd.read_csv(file_path)
# print(housing.head())
# continuous data
housing["median_income"].hist()
# plt.show()
housing["income_cat"] = pd.cut(housing["median_income"],bins=[0.,1.5,3.,4.5,6,np.inf],labels=[1,2,3,4,5])
# housing["income_cat"].hist()
# plt.show()
# print(housing.head())

# stratified sampling
# n_splits: int, default=10
# Number of re-shuffling & splitting iterations.
# test_size: float or int, default=None
# If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.1.
# train_size: float or int, default=None
# If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. If int, represents the absolute number of train samples. If None, the value is automatically set to the complement of the test size.
# print(split)
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=21)

# print(split.split(housing,housing["income_cat"]))
for train_index, test_index in split.split(housing,housing["income_cat"]): #Generate indices to split data into training and test set.
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
print(strat_train_set)
print(strat_test_set)

print(strat_test_set["income_cat"].value_counts())

print(strat_test_set["income_cat"].value_counts()/len(strat_test_set))


# lets drop the income_cat column as we have to stratified sampling data (representing the actual data set)
 
for set_ in (strat_test_set,strat_train_set):
    set_.drop("income_cat",axis=1,inplace=True)

print(strat_test_set)
print(strat_train_set)


