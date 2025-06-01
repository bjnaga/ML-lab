import pandas as pd
import os
import matplotlib.pyplot as plt

file_path = os.path.join("datasets", "housing")
os.makedirs(file_path, exist_ok=True)

file_path_csv = os.path.join(file_path, "housing.csv")
file_path_excel = os.path.join(file_path, "housing.xlsx")

if os.path.exists(file_path_csv):
    housing = pd.read_csv(file_path_csv)
    print("CSV file loaded successfully.")
elif os.path.exists(file_path_excel):
    housing = pd.read_excel(file_path_excel)
    print("Excel file loaded successfully.")
else:
    print("No data file found.")
    housing = pd.DataFrame()


#--------------------------------Exploring----------------------------------------
print(housing.head())
print(housing.info())
print(housing["ocean_proximity"].value_counts())
print(housing.describe())

housing.hist()
housing.hist(bins=50)
housing.hist(bins=50,figsize=(20,15))
plt.show()