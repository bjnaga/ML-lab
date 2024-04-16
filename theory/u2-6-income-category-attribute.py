import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

file_path = os.path.join("datasets","housing","housing.csv")

housing = pd.read_csv(file_path)
# continuous data
housing["median_income"].hist()
plt.show()
housing["income_cat"] = pd.cut(housing["median_income"],bins=[0.,1.5,3.,4.5,6,np.inf],labels=[1,2,3,4,5])
housing["income_cat"].hist()
plt.show()
print(housing.head())




