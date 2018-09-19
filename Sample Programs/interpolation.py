"""
Ilustrates interpolation methods using the scipy library
Programmed by Olac Fuentes
Last modified September 18, 2018
"""
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import interpolate

data_points = 20
method = 'nearest'
#method = 'linear'
#method = 'cubic'
x_data = np.linspace(0, 2*math.pi, num=data_points)
y_data = np.sin(x_data)

x_predict = np.linspace(0, 2*math.pi, num=1000)
y_true = np.sin(x_predict) #Actual value used to evaluate methods, normally not known

# Build interpolator               
f = interpolate.interp1d(x_data, y_data,kind=method)               
                    
#Apply interpolator         
y_predict = f(x_predict)
error = np.mean(np.abs(y_predict-y_true))

plt.figure()
plt.plot(x_data, y_data, '*g',label='data points')
plt.plot(x_predict, y_true, 'b',label='true values')
plt.plot(x_predict, y_predict, 'r',label='predicted values')

title = 'Results using '+method+' and '+str(data_points)+' data points'
plt.title(title)
plt.legend()
plt.show()

print("Method =",method)
print("Data points =",data_points)
print("Mean absolute error = ",error)

