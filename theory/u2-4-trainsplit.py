import numpy as np
import os
import pandas as pd

file_path = os.path.join("datasets","housing","housing.csv")

housing = pd.read_csv(file_path)
# np.random.seed(20)
shuffled_indices = np.random.permutation(len(housing))
print(shuffled_indices)
test_set_size = int(len(housing)* 0.2)
print(test_set_size)
test_indices = shuffled_indices[:test_set_size]
train_indices = shuffled_indices[test_set_size:]
train_set=housing.iloc[train_indices]
test_set=housing.iloc[test_indices]
print(test_set)
print(train_set)