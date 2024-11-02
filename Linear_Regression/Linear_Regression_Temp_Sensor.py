################################################################################################
#
# Example of how to perform simple linear regression, applying it to the adjustment of a curve obtained
# by two temperature sensors. One sensor served as a reference for adjusting the calibration curve
# of another type k sensor.
# 
################################################################################################

# Importing the libraries
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


# Creating the database
data = {'T_Ref': [25.5, 79, 82, 84, 97],
         'T_Sensor_Type_K': [113, 325, 336, 347, 397]}

data = pd.DataFrame(data = data)

# Separating the data:
# T_Ref = X is the independent variable
# T_Sens_Tipo_K = Y is the dependent variable
X = data['T_Ref'].values
Y = data['T_Sensor_Type_K'].values

# Function to transpose the X coordinate values
X = X.reshape(-1, 1)

# Creating the instance for the linear regression class
adjustment = LinearRegression(fit_intercept = True) # y(x) = b + a*x  


# Passing the data to make adjustments
adjustment.fit(X, Y)

# Variables assigned to the least squares (R^2) values ​​resulting from the fits
R_sq = adjustment.score(X, Y)

b = adjustment.intercept_ #adjustment parameter b(Linear Coefficient of the line)
a = adjustment.coef_ # parameter a of adjustment(Angular Coefficient of the line)

print("========================================================")
print(" ** PARAMETERS OBTAINED FOR adjustment1 **")
print(" ========================================================")
print(" Linear Coefficient for adjustment1 (b): ", b)
print(" Angular Coefficient for adjustment1 (a): ", a)
print(" R^2 value for adjustment1: ", R_sq)


# Visualizando o gráfico
fig_size = plt.figure(figsize=(10,6))


plot_adjustment= plt.scatter(X, Y, color = 'blue')
plot_adjustment= plt.plot(X, adjustment.predict(X), color = 'red')
plot_adjustment= plt.title('adjustment')
plot_adjustment= plt.xlabel('T_Ref')
plot_adjustment= plt.ylabel('T_Sensor_Type_K')

plt.show()

