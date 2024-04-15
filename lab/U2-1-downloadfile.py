# downloading raw dataset housing.csv file from github repository

import urllib
import urllib.request
import os

# https://github.com/ageron/handson-ml2/blob/master/datasets/housing/housing.csv
# local_filename, headers = urllib.request.urlretrieve('https://github.com/ageron/handson-ml2/blob/master/datasets/housing/housing.csv')
# html = open(local_filename)
# print(local_filename)
# print(html)
# html.close()
# https://raw.githubusercontent.com/mikolalysenko/lena/master/lena.png
# https://raw.githubusercontent.com/handson-ml2/master/datasets/housing/housing.csv


file_path = os.path.join("datasets","housing")
os.makedirs(file_path,exist_ok=True)
file_path = os.path.join(file_path,"housing.csv")

urllib.request.urlretrieve("https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv", file_path)



