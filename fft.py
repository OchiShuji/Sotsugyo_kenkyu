import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from scipy import interpolate

def b_spline(x,y,deg,N):
    """interpolate with B spline"""
    tck,u = interpolate.splprep([x,y],k=deg,s=0)
    X = np.hstack((np.linspace(1.0,0.0,N),np.linspace(0.0,1.0,N)))
    spline = interpolate.splev(X,tck)
    return spline[0],spline[1]

def arc_len(x,y):
    """calculate arc length"""
    l = np.zeros(len(x))
    for i in range(0,len(x)):
        for j in range(0,i):
            dl = np.sqrt((x[j+1]-x[j])**2+(y[j+1]-y[j])**2)
            l[i] = l[i] + dl
    return l

def input_shape(file_name):
    """input curve shape from csv"""
    data = pd.read_csv(file_name,header=None)
    x_data = data.iloc[:,0]
    y_data = data.iloc[:,1]
    x = np.array(x_data).T
    y = np.array(y_data).T
    return x,y

x_5,y_5 = input_shape("/Users/ochihideji/Desktop/a1_卒業研究/5.csv")
x_20,y_20 = input_shape("/Users/ochihideji/Desktop/a1_卒業研究/20.csv")
x_naca,y_naca = input_shape("/Users/ochihideji/Desktop/a1_卒業研究/naca0015.csv")

N = 512

x_5_spl,y_5_spl = b_spline(x_5,y_5,3,N)
plt.plot(x_5_spl,y_5_spl)
x_20_spl,y_20_spl = b_spline(x_20,y_20,3,N)
plt.plot(x_20_spl,y_20_spl)
x_naca_spl,y_naca_spl = b_spline(x_naca,y_naca,3,N)
plt.plot(x_naca_spl,y_naca_spl)
plt.show()


arc_length_5 = arc_len(x_5_spl,y_5_spl)
plt.plot(arc_length_5,y_5_spl,label="5")
arc_length_20 = arc_len(x_20_spl,y_20_spl)
plt.plot(arc_length_20,y_20_spl,label="20")
arc_length_naca = arc_len(x_naca_spl,y_naca_spl)
plt.plot(arc_length_naca,y_20_spl,label="naca")
plt.legend()
plt.show()

F_5 = np.fft.fft(y_5_spl)
amp_5 = np.abs(F_5/(N/2))
freq_5 = np.fft.fftfreq(2*N)
plt.plot(freq_5[:N],amp_5[:N],label="5")
print(freq_5[:N])


F_20 = np.fft.fft(y_20_spl)
amp_20 = np.abs(F_20/(N/2))
freq_20 = np.fft.fftfreq(2*N)
plt.plot(freq_20[:N],amp_20[:N],label="20")
print(freq_20[:N])


plt.legend()
plt.show()

plt.plot(freq_20[:N],amp_20[:N],label="20")

F_naca = np.fft.fft(y_naca_spl)
amp_naca = np.abs(F_naca/(N/2))
freq_naca = np.fft.fftfreq(2*N)
plt.plot(freq_naca[:N],amp_naca[:N],label="naca")
print(freq_naca[:N])

plt.legend()
plt.show()





