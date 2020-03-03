# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 08:55:04 2020

@author: clyde
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#load spectra data
helium, wave_he = np.loadtxt(r'C:\Users\clyde\OneDrive\Documents\Python\spectra lab\spectra.txt', skiprows = 1, unpack = True)
hydrogen = np.loadtxt(r'C:\Users\clyde\OneDrive\Documents\Python\spectra lab\hydrogen.txt', skiprows = 1, unpack = True)
unknown = np.loadtxt(r"C:\Users\clyde\OneDrive\Documents\Python\spectra lab\unknown.txt", skiprows = 1, unpack = True)

wave_he = wave_he*10**(-9)
lambda_0 = 282.8*10**(-9)
inv_wave = 1/(wave_he-lambda_0)

uncert_he = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
uncert_lambda0 = 0.4*10**(-9)
uncert_he = -1*(helium**2)*uncert_he

#define Hartman Equation
def f(x,m,b):
    hart = (m/(x-lambda_0))+b
    return hart
#define Balmer Equation
def g(x, r):
    balmer = r*((1/4)-(1/np.power(x,2)))
    return balmer

#Find parameters m and b
popt, pcov = curve_fit(f,wave_he, helium, sigma = uncert_he)
perr = np.sqrt(np.diag(pcov))
wave_hy = (popt[0]/(hydrogen-popt[1]))
wave_unk = (popt[0]/(unknown-popt[1]))


wave_hy = 1/wave_hy
wave_unk = 1/wave_unk

plt.figure(1)
plt.scatter(inv_wave, helium)
plt.plot(inv_wave, f(wave_he, *popt))
plt.errorbar(inv_wave, helium, yerr = uncert_lambda0, linestyle='None')
plt.title('Scale reading vs. Wavelength')
plt.xlabel('Wavelength (cm)')
plt.ylabel('Scale Reading (cm)')
plt.show()

mean_h = np.sum(helium)/len(helium)
var_h = np.sum((helium-mean_h)**2)/3
chi = np.sum((helium-f(wave_he, *popt))**2/var_h)
print('reduced chi square value is:', round(chi,4))

#scale = input('Enter a value for the scale reading (cm): ')
#scale = float(scale)
#wavelength = (popt[0]/(scale-popt[1]))+lambda_0
#print('The wavelength is: ', wavelength, 'nm')

n = [3,4,5,6]
uncert_hy = [0.01,0.01,0.01,0.01]
wave_hy = (popt[0]/(hydrogen-popt[1]))+lambda_0
#uncert_wavehy = (uncert_hy**2+uncert_lambda0**2)**(1/2)
wave_hy = 1/wave_hy
uncert_wavehy = -1*(wave_hy**-2)*uncert_hy

popt2, pcov2 = curve_fit(g,n,wave_hy, sigma = uncert_wavehy)
perr2 = np.sqrt(np.diag(pcov2))

plt.figure(2)
plt.scatter(n,wave_hy)
plt.plot(n, g(n,*popt2))
plt.errorbar(n,wave_hy, yerr = uncert_wavehy, linestyle='None')
plt.title('Wavelength vs. Energy Level')
plt.xlabel('Energy Level')
plt.ylabel('Inverse Wavelength')
plt.show()

mean_c = np.sum(wave_hy)/len(wave_hy)
var_c = np.sum((wave_hy-mean_c)**2)/3
chi = np.sum((wave_hy-g(n, *popt2))**2/var_c)
print('reduced chi square value is:', round(chi,4))

h = 6.62*10**(-34)
c = 3*10**(8)
print('Value of hcR obtained was: ', c*h*popt2[0]*(6.24*10**18), 'eV')
