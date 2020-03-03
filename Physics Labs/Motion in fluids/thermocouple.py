# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 20:03:59 2020

@author: clyde
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#define Seebeck Equation
def f(x,s,t):
    emf = s*(x-t)
    return emf

t = 6 #temp of ice bath in degrees Celsius
emf, temp = [] #load data from txt document


voltage = input("Enter the voltage you have measure in Volts: ")
temperature = voltage
print('The temperature reached is: ', temperature)