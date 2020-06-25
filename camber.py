import pandas as pd
import numpy as np 
from scipy import interpolate
from matplotlib import pyplot as plt 

case = input(">>case?")
df = pd.ExcelFile('/Users/ochihideji/Desktop/AA_航空/4s/a1_卒業研究/cad_変形データ.xlsx')
sheet_df = df.parse("case"+case,header=7)

def max_camb(flg):
    if flg==1:
        l = 9
    elif flg==2:
        l = 14
    elif flg==3:
        l = 17
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

print("==case"+case+"==")
for i in range(1,4):
    print("\nforce"+str(i)+":")
    print(max_camb(i))

