# purely random sampling
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

file_path = os.path.join("datasets","housing","housing.csv")

housing = pd.read_csv(file_path)
train_set , test_set = train_test_split(housing,test_size=0.2,random_state=21)
print(train_set,test_set)
