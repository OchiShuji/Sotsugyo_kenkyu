import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from scipy import interpolate

def b_spline(x,y,deg,N):
    """interpolate with B spline"""
    tck,u = interpolate.splprep([x,y],k=deg,s=0)
    X = np.linspace(1.0,0.0,N)
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

def discrete_plot(x,y,N):
    x = x.tolist()
    y = y.tolist()
    f = interpolate.interp1d(x,y,kind="cubic")
    T = x[-1]
    dt = T / N
    x_new = np.linspace(0,T,N)
    y = f(x_new)
    return y,dt

"""input shape"""
x_5,y_5 = input_shape("/Users/ochihideji/Desktop/a1_卒業研究/5.csv")
x_20,y_20 = input_shape("/Users/ochihideji/Desktop/a1_卒業研究/20.csv")
x_naca,y_naca = input_shape("/Users/ochihideji/Desktop/a1_卒業研究/naca0015.csv")
x_ex,y_ex = input_shape("/Users/ochihideji/Desktop/a1_卒業研究/実験/201015_test/sortspline/sort_10_1_mod.csv")
x_ex,y_ex = x_ex/2000,y_ex/2000

N = 1024

"""interpolate shape"""
x_5_spl,y_5_spl = b_spline(x_5,y_5,3,N)
plt.plot(x_5_spl,y_5_spl,label="5")
x_20_spl,y_20_spl = b_spline(x_20,y_20,3,N)
plt.plot(x_20_spl,y_20_spl,label="20")
x_naca_spl,y_naca_spl = b_spline(x_naca,y_naca,3,N)
plt.plot(x_naca_spl,y_naca_spl,label="naca")
x_ex_spl,y_ex_spl = b_spline(x_ex,y_ex,3,N)
plt.plot(x_ex_spl,y_ex_spl,label="experiment20201015")
plt.legend()
plt.ylabel("y/c")
plt.xlabel("x/c")
plt.show()

"""plot y as the function of arch length"""
arc_length_5 = arc_len(x_5_spl,y_5_spl)
plt.plot(arc_length_5,y_5_spl,label="5")
arc_length_20 = arc_len(x_20_spl,y_20_spl)
plt.plot(arc_length_20,y_20_spl,label="20")
arc_length_naca = arc_len(x_naca_spl,y_naca_spl)
plt.plot(arc_length_naca,y_naca_spl,label="naca")
arc_length_ex = arc_len(x_ex_spl,y_ex_spl)
plt.plot(arc_length_ex,y_ex_spl,label="experiment20201015")
plt.legend()
plt.ylabel("y/c")
plt.xlabel("archlength/c")
plt.show()

f_5,dt_5 = discrete_plot(arc_length_5,y_5_spl,N)
f_20,dt_20 = discrete_plot(arc_length_20,y_20_spl,N)
f_naca,dt_naca = discrete_plot(arc_length_naca,y_naca_spl,N)
f_ex,dt_ex = discrete_plot(arc_length_ex,y_ex_spl,N)

F_5 = np.fft.fft(f_5)
amp_5 = np.abs(F_5/(N/2))
freq_5 = np.fft.fftfreq(2*N,d=dt_5)
plt.plot(freq_5[:N],amp_5[:N],label="5")

F_20 = np.fft.fft(f_20)
amp_20 = np.abs(F_20/(N/2))
freq_20 = np.fft.fftfreq(2*N,dt_20)
plt.plot(freq_20[:N],amp_20[:N],label="20")

F_ex = np.fft.fft(f_ex)
amp_ex = np.abs(F_ex/(N/2))
freq_ex = np.fft.fftfreq(2*N,d=dt_ex)
plt.plot(freq_ex[:N],amp_ex[:N],label="experiment")

F_naca = np.fft.fft(f_naca)
amp_naca = np.abs(F_naca/(N/2))
freq_naca = np.fft.fftfreq(2*N,d=dt_naca)
plt.plot(freq_naca[:N],amp_naca[:N],label="naca")

plt.xlim(0,5)
plt.ylim(0,0.1)
plt.xlabel("frequency[Hz]")
plt.ylabel("amplitude")
ax = plt.gca()
ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.01))
ax.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
ax.grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.75')
ax.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
ax.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.75')
plt.legend()
plt.show()


"""
#test 
dt = 0.001
t = np.linspace(0,N*dt,N)
s = np.sin(2*np.pi*50*t)+np.sin(2*np.pi*120*t)

F_s = np.fft.fft(s)
freq_s = np.fft.fftfreq(N,dt)
amp_s = np.abs(F_s/(N/2))
plt.plot(freq_s[0:int(N/2)],amp_s[0:int(N/2)])
plt.show()
"""