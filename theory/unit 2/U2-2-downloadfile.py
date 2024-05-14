# downloading raw dataset housing.tgz file from github repository

import urllib
import urllib.request
import os
import tarfile

file_path = os.path.join("datasets","housing")
os.makedirs(file_path,exist_ok=True)
file_full_path = os.path.join(file_path,"housing.tgz")

urllib.request.urlretrieve("https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz", file_full_path)

housing_tgz = tarfile.open(file_full_path)
housing_tgz.extractall(path=file_path)
housing_tgz.close()
