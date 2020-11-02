import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from scipy import interpolate
from os import path
import glob
import csv


file_path = "/Users/ochihideji/Desktop/a1_卒業研究/実験/201028_test/csv"
files = []
for obj in glob.glob("{}/cambersp*.csv".format(file_path)):
    if path.isfile(obj):
        files.append(obj)
    else:
        pass
files = sorted(files)

def input_shape(file_name):
    """input curve shape from csv"""
    data = pd.read_csv(file_name,header=None)
    x_data = data.iloc[:,0]
    y_data = data.iloc[:,1]
    x = np.array(x_data).T
    y = np.array(y_data).T
    return x,y

f = open("{}/output.csv".format(file_path),"w")
w = csv.writer(f)
w.writerow(["file_path","max_camber_x","max_camber_y"])
for i in range(0,len(files)):
    camber_x,camber_y = input_shape(files[i])
    max_camb_y = max(camber_y)
    max_camb_x = camber_x[np.argmax(camber_y)]
    w.writerow([files[i],max_camb_x,max_camb_y])
f.close()



    
