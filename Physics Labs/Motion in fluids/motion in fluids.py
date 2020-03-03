# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 20:14:10 2020

@author: clyde
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import statistics

d_high = [1,2,3,4,6] #diameter of sphere
d_low = [1,2,3,5,6] #diameter of sphere

r_high = [0.5,1,1.5,2,3]
r_low = [0.5,1,1.5,2.5,3]

D = 10.5*10 #dimension of container perpendicular to fall
dr_low = np.zeros(5)
dr_high = np.zeros(5)

for i in range(len(d_high)):
    y = d_high[i]/D
    dr_high[i] = y

for i in range(len(d_high)):
    x = d_low[i]/D
    dr_low[i] = x

#load velocity
v_lowr = np.loadtxt(r'C:\Users\clyde\OneDrive\Documents\Python\v_low.txt', skiprows=1, unpack=True)
v_highr = np.loadtxt(r'C:\Users\clyde\OneDrive\Documents\Python\v_highr.txt', skiprows=1, unpack=True)

#uncertainty calculations
pos_unc = 0.5
time_unc = 0.005
time_unc = ((time_unc)**2+(time_unc)**2)**(1/2)
pos_unc = ((pos_unc)**2+(pos_unc)**2)**(1/2)
vel_unc = ((time_unc/0.05)**2+(pos_unc/4.37)**2)**(1/2)
velL_unc = [i*vel_unc for i in v_lowr]
velH_unc = [i*vel_unc for i in v_highr]

vl_corr = np.zeros(5)
vh_corr = np.zeros(5)

#wall effect corrections
for i in range(len(d_high)):
    y = v_lowr[i]/(1-2.109*(dr_low[i])+2.089*(dr_low[i])**2)
    vl_corr[i] = y
for i in range(len(d_high)):
    x = v_highr[i]/(1-2.109*(dr_high[i])+2.089*(dr_high[i])**2)
    vh_corr[i] = x

def f(x,a):
    return a*np.power(x,2) #low reynolds number fit

def g(x,b):
    return b*np.power(x,(1/2)) #high reynolds number fit

def h(x,a,b):
    return np.add(np.multiply(x,a),b) #adjusted high reynolds number fit

#curve fit functions
popt,pcov = curve_fit(f, r_low, vl_corr, sigma = velL_unc)
popt2,pcov2 = curve_fit(g, r_high, vh_corr, sigma =velH_unc)
popt3,pcov3 = curve_fit(h, r_high, vh_corr, sigma =velH_unc)

#print plots
plt.figure(1)
plt.title('Low Reynolds Number: Radius vs. Terminal Velocity')
plt.scatter(r_low,vl_corr, label= 'Observations')
plt.plot(r_low, f(r_low,popt[0]), label = 'Fitted Data')
plt.errorbar(r_low,vl_corr,yerr = velL_unc, linestyle ="None")
plt.xlabel('Radius (mm)')
plt.ylabel('Velocity (mm/s)')
plt.legend(loc='upper left')
plt.savefig('low_reynolds.png')


plt.figure(2)
plt.title('High Reynolds Number: Radius vs. Mean Velocity w/ Radical Fit')
plt.scatter(r_high, vh_corr,label= 'Observations')
plt.plot(r_high, f(r_high,popt2[0]),label = 'Fitted Data')
plt.errorbar(r_high,vh_corr,yerr = velH_unc, linestyle ="None")
plt.xlabel('Radius (mm)')
plt.ylabel('Velocity (mm/s)')
plt.legend(loc='upper left')
plt.savefig('high_reynolds.png')

plt.figure(3)
plt.title('High Reynolds Number: Radius vs. Mean Velocity w/ Linear Fit')
plt.scatter(r_high, vh_corr,label= 'Observations')
plt.plot(r_high, h(r_high,*popt3),label = 'Fitted Data')
plt.errorbar(r_high,vh_corr,yerr = velH_unc, linestyle ="None")
plt.xlabel('Radius (mm)')
plt.ylabel('Velocity (mm/s)')
plt.legend(loc='upper left')
plt.savefig('high_reynolds2.png')
plt.tight_layout()

#variances are calculated
var_l = statistics.variance(vl_corr)
var_h = statistics.variance(vh_corr)

#reduced chi-squared values
chi_l = (np.sum((vl_corr-f(r_low,*popt))**2))/var_l
print('The reduced chi squared value is:', round(chi_l,4))
chi_h = (np.sum((vh_corr-g(r_high,*popt2))**2))/var_h
print('The reduced chi squared value is:', round(chi_h,4))
chi_h2 = (np.sum((vh_corr-h(r_high,*popt3))**2))/var_h
print('The reduced chi squared value is:', round(chi_h2,4))
print()

#%%
#calculate Reynolds numbers

#viscosity in g/mm*s
vis_w = 0.1
vis_gly = 0.934

#cross sectional area
A_low = np.pi*np.power(r_low,2)
A_high = np.pi*np.power(r_high,2)

#viscous force calculation
gly_vis_force = vis_gly*A_low*vl_corr/d_low
wat_viscous_force = vis_w*A_high*vh_corr/d_high

#density of spheres
p_tf = 0.0022
p_ny = 0.00112
p_gly = 0.00126
p_wat = 0.001

#volume of spheres
vol_tf = (4/3)*np.pi*np.power(r_low,3)
vol_ny =(4/3)*np.pi*np.power(r_high,3)

#mass of spheres
m_tf = (p_tf-p_gly)*vol_tf
m_ny = np.power(vol_ny,(p_ny-p_wat))

low_dragf = 6*np.pi*vis_gly*vl_corr*r_low
high_dragf = m_ny*9800

#low reynolds number
low_rey = low_dragf/gly_vis_force
print('The low Reynolds number is calculated to be: ', low_rey)

#high reynolds number
high_rey = high_dragf/wat_viscous_force
print('The high Reynolds number is calculated to be: ', high_rey)

#%%
r = 50 #radius of aluminum sphere
p = 1.26 #density of glycerin
vis = 1500*10**(-2) #viscosity of gly
vol = (4/3)*np.pi*r**3 #vol of sphere
m = p*vol #mass of object
g = 980 #acceleration of gravity
v_t = (((m*g)/(6*np.pi*vis*r))*(1-(1/np.exp(1))))/0.634 #terminal velocity
t = m/(6*np.pi*vis*r) #time constant
print()
print('The time constant assoicated with the Aluminum sphere is: ', round(t,3), 's')
print('The terminal velocity is: ', round(v_t,3), 'cm/s')