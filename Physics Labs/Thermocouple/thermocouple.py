import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

temp, voltage, resistance = np.loadtxt(r'C:\Users\clyde\OneDrive\Documents\Python\Thermocouple\Thermocouple.txt', skiprows =1, unpack = True)

#uncertainties (error in percision)
uncert_v = [0.1,0.1,0.1,0.1,0.1,0.1,0.1]
uncert_t = [0.5, 0.5,0.5, 0.5,0.5, 0.5,0.5]
uncert_r = [0.1,0.1,0.1,0.1,0.1,0.1,0.1]

to = 6+273.15 #temperature of water bath

def f(x,a):
    y = a*(x-to)
    return y

t = temp+273.15 #convert temperature into kelvin
popt, pcov = curve_fit(f, t,voltage, sigma=uncert_v) #least square parameters

#create plot
plt.figure()
plt.scatter(t, voltage, label = 'Experimental Data') #experimental data
plt.plot(t, f(t,popt[0]), color = 'orange', label ='Fitted Data') #fitted data
plt.errorbar(t, voltage, yerr=uncert_v, linestyle = 'None') #errorbars
plt.xlabel('Temperature (K)')
plt.ylabel('Voltage/EMF (V)')
plt.title('Seebeck Effect: Relationship of Voltage and Temperature')
plt.legend()
plt.show()

#reduced chi square calculation/output
mean_v = np.sum(voltage)/len(voltage) #mean Calculation
var_v = np.sum((voltage-mean_v)**2)/2 #Variance calculation
chi = np.sum((voltage-f(t, *popt))**2/var_v) #reduced chi squared 
print('Reduced chi square value is:', round(chi,4))
print('The Seebeck coefficient is: ', round(popt[0],4), 'V/K')

#Estimate temperaure for inputted voltage
emf = input("Enter a value for EMF (V): ")
emf = float(emf)
temperature = ((emf/popt[0])+to)-273.15
print('The estimated temperature is: ', round(temperature,1), 'degrees Celsius')