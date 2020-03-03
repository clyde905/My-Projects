# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:29:54 2020

@author: clyde
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import statistics as sta

#Load Data for different weights
curr_0, post_0 = np.loadtxt(r'0-1.txt',skiprows=1,usecols=(0,2), unpack=True)
curr_550, post_550 = np.loadtxt(r'550-1.txt',skiprows=1,usecols=(0,2),unpack=True)
curr_1100, post_1100 = np.loadtxt(r'5502-1.txt',skiprows=1,usecols=(0,2), unpack=True)
curr_0 = curr_0[1:9]
post_0 = post_0[1:9]
#Define constants
g,d,l = 9.8, 0.14, 0.25

#convert position to meters
post_0, post_550, post_1100 =  post_0*10**(-2), post_550*10**(-2), post_1100*10**(-2)
#square current values
curr_0_sq, curr_550_sq, curr_1100_sq = curr_0**2, curr_550**2, curr_1100**2

mass = np.array([0,550,1100]) # mass of loads
mass = mass*10**(-6) #convert mass to kg

#find force applied by weights
f_grav = []
for i in range(len(mass)):
    force = mass[i]*g
    f_grav.append(force)

#current uncertainties
'uncertainty of current measurements is 0.5A'
def unc(x):
    return (x**2)*2*(0.5/x)

curr_unc0 = unc(curr_0)
curr_unc550 = unc(curr_550)
curr_unc1100 = unc(curr_1100)


#curve fit function
def f(x,a,b):
    return a*x+b

pop, cov = curve_fit(f,post_0,curr_0_sq)
pop2, cov2 = curve_fit(f,post_550,curr_550_sq)
pop3, cov3 = curve_fit(f,post_1100,curr_1100_sq)

plt.figure(1)
plt.scatter(post_0, curr_0_sq, label = 'Experimental Data')
plt.plot(post_0, f(post_0,*pop), label = 'Fitted Data')
plt.errorbar(post_0,curr_0_sq,yerr= curr_unc0,label='Error bars',linestyle='None')
plt.xlabel('Seperation Distance (m)')
plt.ylabel('Squared Current (A^2)')
plt.title('Distance between Wires W.R.T Change in Current: Unloaded')
plt.legend()
plt.savefig('currentbalance0.png')

plt.figure(2)
plt.scatter(post_550, curr_550_sq, label = 'Experimental Data')
plt.plot(post_550, f(post_550,*pop2), label = 'Fitted Data')
plt.errorbar(post_550,curr_550_sq,yerr= curr_unc550,label='Error bars',linestyle='None')
plt.xlabel('Seperation Distance (m)')
plt.ylabel('Squared Current (A^2)')
plt.title('Distance between Wires W.R.T Change in Current: 550mg weight')
plt.legend()
plt.savefig('currentbalance550.png')

plt.figure(3)
plt.scatter(post_1100, curr_1100_sq, label = 'Experimental Data')
plt.plot(post_1100, f(post_1100,*pop3), label = 'Fitted Data')
plt.errorbar(post_1100,curr_1100_sq,yerr= curr_unc1100, label='Error bars', linestyle='None')
plt.xlabel('Seperation Distance (m)')
plt.ylabel('Squared Current (A^2)')
plt.title('Distance between Wires W.R.T Change in Current: 1100mg weight')
plt.legend()
plt.savefig('currentbalance1100.png')

'''
"Since formula (1) is accurate only when the
distance between the wires is much larger than their diameter, take points where 
the distance between the wires is relatively large". Therefore, we choose the last points
on each array to calculate the magnetic permeability
'''

current_sq = [curr_0_sq[7],curr_550_sq[8], curr_1100_sq[8]]
pos = [post_0[7], post_550[8], post_1100[8]]

def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

mu = []
for i in range(1,3):
    m = (f_grav[i]/l)*(2*np.pi*pos[i]/current_sq[i])
    mu.append(m)
print('\nFor the 550mg and 1100mg weights the values of the mu corresponding to an increase \
 in weight are: ', format_e(mu[0]), " and ", format_e(mu[1]))

#uncertainty in constant
pos_unc = np.float(0.05/100)
curr_unc = [curr_unc550[8],curr_unc1100[8]]
mu_uncert = []
for i in range(len(mu)):
    mu_unc = (((curr_unc[i]/current_sq[i])**2+(pos_unc/pos[i])**2)**(1/2))*mu[i]
    mu_uncert.append(mu_unc)

#check relationship between force and distance between the wires: F= 1/d
for i in range(1,3):
    force_percent = ((np.abs(f_grav[i]-(1/pos[i])))/(np.mean((f_grav[i],(1/pos[i])))))*100
    print('\nThe percent difference between force ',i,  ' is: ', round(force_percent,2), '%')

#percent difference for mu values from real value    
mu_true = float(4*np.pi*10**(-7))
ave1= (mu_true+mu[0])/2
ave2 = (mu_true+mu[1])/2
average = [ave1, ave2]
diff=np.zeros(2)
for i in range(len(mu)):
    diff[i] = ((np.abs(mu_true-mu[i]))/average[i])*100

print('\nFor the 550mg and 1100mg weights, the percentage differences\
 between the\ncalculated mu value and the actual value are: ', round(diff[0],2), \
'% and ', round(diff[1],2), '%, \nrespectively')

'reduced chi squared Calculations'
var0, var550, var1100 = np.var(curr_0_sq), np.var(curr_550_sq),np.var(curr_1100_sq)
chi0 = np.sum((curr_0_sq-f(post_0, *pop))**2/var0)
chi550 = np.sum((curr_550_sq-f(post_550, *pop2))**2/var550)
chi1100 = np.sum((curr_1100_sq-f(post_1100, *pop3))**2/var1100)

print('\nThe Reduced Chi Square value for the unloaded measurements is: ', round(chi0,2))
print('The Reduced Chi Square value for the 550mg weight measurements is: ', round(chi550,2))
print('The Reduced Chi Square value for the 1100mg weight measurements is: ', round(chi1100,2))