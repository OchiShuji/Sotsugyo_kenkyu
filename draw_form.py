import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from os import path
import os
import glob
import csv
from scipy import interpolate

cwd = os.getcwd()
file_path = os.path.join(cwd,"input")
files = []
for obj in glob.glob("{}/*.csv".format(file_path)):
    if path.isfile(obj):
        files.append(obj)
    else:
        pass
files = sorted(files)
f = open("{}/title.txt".format(file_path),"r")
title = f.read()

def input_shape(file_name):
    """input curve shape from csv"""
    data = pd.read_csv(file_name,header=None)
    x_data = data.iloc[:,0]
    y_data = data.iloc[:,1]
    x = np.array(x_data).T
    y = np.array(y_data).T
    return x,y

def b_spline(x,y,deg,N):
    """interpolate with B spline"""
    tck,u = interpolate.splprep([x,y],k=deg,s=0)
    X = np.linspace(1.0,0.0,N)
    spline = interpolate.splev(X,tck)
    return spline[0],spline[1]

def plot(x,y,l):
    textsize = 10
    plt.tick_params(labelsize = textsize)
    plt.xlabel("x/c",fontsize = textsize)
    plt.ylabel("y/c",fontsize = textsize)
    ind = l[0]
    
    """
    if ind == "0":
        plt.plot(x,y,label=l[2:],c="red",linewidth=1.5,zorder=6)
    if ind == "1":
        plt.plot(x,y,label=l[2:],c="green",linewidth=1)
    else:
        plt.plot(x,y,c="green",linewidth=1)
    """
    
    if ind == "1":
        plt.plot(x,y,label=l[2:],c="red")
    if ind == "2":
        plt.plot(x,y,label=l[2:],c="red",linestyle="--",linewidth=1)
    if ind == "3":
        plt.plot(x,y,label=l[2:],c="green")
    if ind == "4":
        plt.plot(x,y,label=l[2:],c="green",linestyle="--",linewidth=1)

    """
    if ind == "1":
        plt.plot(x,y,label=r"$\Delta$x/c=0.05")
    if ind == "2":
        plt.plot(x,y,label=r"$\Delta$x/c=0.1")
    if ind == "3":
        plt.plot(x,y,label=r"$\Delta$x/c=0.2")
    """
    """
    plt.plot(x,y,label=l[2:])
    """
    plt.grid(True)
    plt.axes().set_aspect('equal')
    plt.ylim(-0.12,0.12)
    plt.xlim(0.3, 1.0)
    plt.legend(loc="lower right",ncol=3,fontsize=15)

df_ref = pd.read_csv("/Users/ochihideji/Desktop/a1_卒業研究/naca0015.csv",header=None)
ref = np.array(df_ref).T
plt.figure(figsize=(15, 7))
plt.title(title,fontsize=20)

plt.plot(ref[0,:],ref[1,:],linestyle="--",linewidth=1,label="NACA0015",c="grey")
plt.plot(0.3,0,label=" ",c="white")
for obj in files:
    l = os.path.splitext(os.path.basename(obj))[0]
    x,y = input_shape(obj)
    x_sp,y_sp = b_spline(x,y,3,1000)
    plot(x_sp,y_sp,l)

plt.savefig(os.path.join(cwd,"{}.png".format(title)))
plt.show()

