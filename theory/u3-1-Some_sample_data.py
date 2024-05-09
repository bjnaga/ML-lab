import mglearn
import numpy as np
import matplotlib.pyplot as plt

X,y = mglearn.datasets.make_forge()
print(X,y)
# print(X[:,0])
mglearn.discrete_scatter(X[:,0],X[:,1],y)
# scatter plot of forge dataset
plt.legend(["Class 0","Class 1"],loc=5)
plt.xlabel("First feature")
plt.ylabel("Second label")
plt.show()
print("X.shape:{}".format(X.shape))

# scatter plot of wave dataset

X,y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X,y,'o')
plt.ylim(-3,3)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()

# Wisconsin Breast cancer dataset 

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() # bunch objects
print(cancer.keys())
print(cancer.target_names)
print("Shape of the cancer data:{}".format(cancer.data.shape))
print(np.bincount(cancer.target))
# print("Sample counts per class : \n {}".format({n: v} for n, v in zip(cancer.target_names,np.bincount(cancer.target))))
print("Features names of data {}".format(cancer.feature_names))
print("about Data ",cancer.DESCR)


# Real World Regression dataset Boston Housing Dataset 1970's  506 X 13
 
# from sklearn.datasets import load_boston
# boston =load_boston()
# print("data shape {}".format(boston.data.shape))

# Alternative boston download 
# import pandas as pd
# import numpy as np

# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# target = raw_df.values[1::2, 2]
# print("BOSTON data",data)
# print("Boston Targets",target)

# from sklearn.datasets import fetch_openml
# housing = fetch_openml(name="house_prices", as_frame=True)
# print(housing.shape)

from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
print(diabetes.keys())
print(diabetes.data.shape)
print(diabetes.target)



X,y = mglearn.datasets.load_extended_boston()
print("X.shape :{}".format(X.shape))
