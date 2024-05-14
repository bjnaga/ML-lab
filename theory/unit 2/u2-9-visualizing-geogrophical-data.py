import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
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

print(strat_test_set["income_cat"].value_counts())

print(strat_test_set["income_cat"].value_counts()/len(strat_test_set))


# lets drop the income_cat column as we have to stratified sampling data (representing the actual data set)
 
for set_ in (strat_test_set,strat_train_set):
    set_.drop("income_cat",axis=1,inplace=True)
print(strat_test_set)
print(strat_train_set)

# program starts here
housing = strat_train_set.copy()
print(housing)
housing.plot(kind="scatter",x="longitude",y="latitude")
housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1)

plt.show()


# housing prices with tadius of each circle represents the district's population (options s), 
# and the color represents the price (option c). We use predefined color map(option cmap) called 'jet', 
# which ranges from blue(low price) to red (high price)

# housing.plot(kind="scatter",x="longitude",y="latitude", alpha=0.4,s=housing["population"]/100,label ="population",figsize=(10,7)
#     ,c="median_house_value",cmap=plt.get_cmap("jet"),
#     # colorbar=True,
# )
housing.plot(kind="scatter",x="longitude",y="latitude", alpha=0.4,s=housing["population"]/100,label ="population",figsize=(10,7)
    ,c="median_house_value",cmap=plt.get_cmap("jet"),
    colorbar=True,
)
plt.show()

# looking for corelation - standard corelation coefficient (Pearsons's r)
# Method of correlation:
# pearson : standard correlation coefficient
# kendall : Kendall Tau correlation coefficient
# spearman : Spearman rank correlation
# callable: callable with input two 1d ndarrays
# and returning a float. Note that the returned matrix from corr will have 1 along 
# the diagonals and will be symmetric regardless of the callable’s behavior.

corr_matrix = strat_train_set.corr(method="pearson",numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))
pd.plotting.scatter_matrix(housing)
print(housing)
plt.show()
attribute =  ["median_house_value","median_income","total_rooms","housing_median_age"]
pd.plotting.scatter_matrix(housing[attribute])
print(housing)
plt.show()

# lets focus zoom in on two attribute that seems to have some corelation 

housing.plot(kind="scatter",x="median_income",y="median_house_value",alpha=0.1)
plt.show()

# experimenting with attribute combinations
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

corr_matrix = housing.corr(method="pearson",numeric_only=True)

print(corr_matrix["median_house_value"].sort_values(ascending=False))
