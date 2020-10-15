import pandas as pd
import numpy as np 
from scipy import interpolate
from matplotlib import pyplot as plt 
import csv

case = input(">>case? \n")
ref = input(">>refinement? - y/n \n")
df = pd.ExcelFile('/Users/ochihideji/Desktop/a1_卒業研究/cad_変形データ_分布.xlsx')
sheet_df = df.parse("case"+case,header=7)


def max_camb(flg):
    """flg：同ケース内の別イタレーション"""
    if flg==1:
        l = 9
    elif flg==2:
        l = 14
    elif flg==3:
        l = 19
    elif flg==4:
        l = 24
    elif flg==5:
        l = 29
    camber_df = sheet_df.iloc[:24,l]
    x_df = sheet_df.iloc[:24,3]
    camber = np.array(camber_df).T
    x = np.array(x_df).T
    f = interpolate.interp1d(x,camber,kind="cubic") 
    x_fine = np.linspace(min(x),max(x),1000)
    plt.plot(x_fine,f(x_fine))
    plt.show()
    m = max(f(x_fine))
    p = x_fine[np.where(f(x_fine)==m)]
    return p,m

def naca0015():
    x_df = sheet_df.iloc[:47,1]
    y_df = sheet_df.iloc[:47,2]
    x = np.array(x_df).T
    y = np.array(y_df).T
    return x,y

def morph(flg):
    if flg==1:
        k = 8
    elif flg==2:
        k = 13
    elif flg==3:
        k = 18
    elif flg==4:
        k = 23
    elif flg==5:
        k = 28
    x_df = sheet_df.iloc[:47,3]
    y_df = sheet_df.iloc[:47,k]
    x = np.array(x_df).T / 200
    y = np.array(y_df).T / 200
    return x,y


def refinement(x,y):
    tck,u = interpolate.splprep([x,y],s=0)
    theta = np.linspace(0,np.pi,50)
    c = x.max()-x.min()
    x_ref = 0.5*c + 0.5*c*np.cos(theta)
    spline = interpolate.splev(x_ref,tck)
    return spline[0],spline[1]

print("==case"+case+"==")
for i in range(1,6):
    print("\nIteration#"+str(i)+":")
    print(max_camb(i))

if ref == "y":
    x,y = naca0015()
    x_ref,y_ref = refinement(x,y)
    print(len(x_ref),len(y_ref))
    plt.scatter(x_ref,y_ref)
    plt.show()
    f = open("/Users/ochihideji/Desktop/a1_卒業研究/code/naca0015.dat","w")
    w = csv.writer(f)
    for i in range(0,len(x_ref)):
        w.writerow([np.round(x_ref[i],4),np.round(y_ref[i],4)])
    f.close()
    xm,ym = morph(1)
    xm_ref,ym_ref = refinement(xm,ym)
    plt.scatter(xm_ref,ym_ref)
    plt.scatter(x_ref,y_ref)
    plt.show()
    f = open("/Users/ochihideji/Desktop/a1_卒業研究/code/morph.dat","w")
    w = csv.writer(f)
    for i in range(0,len(xm_ref)):
        w.writerow([np.round(xm_ref[i],4),np.round(ym_ref[i],4)])
    f.close()



