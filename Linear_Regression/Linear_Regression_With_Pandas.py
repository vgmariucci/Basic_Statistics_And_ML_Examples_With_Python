################################################################################################
#
# Example of how to perform simple linear regression, with emphasis on comparing
# the least squares R^2 performance metric or parameter of each fit
# 
################################################################################################

# Importing the libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


# Creating the database
data = {'X': [5, 15, 25, 35, 45, 55],
         'Y': [5, 20, 14, 32, 22, 38]}

data = pd.DataFrame(data = data)

# Separating the data:
# X is the independent variable
# Y is the dependent variable
X = data['X'].values
Y = data['Y'].values

# Function to transpose the X coordinate values
X = X.reshape(-1, 1)

# Creating the instance for the linear regression classes
adjustment1  = LinearRegression(fit_intercept= False) # y(x) = b1*x  (b0 = 0)
adjustment2  = LinearRegression(fit_intercept= True)  # y(x) = b0 + b1*x

# Passing the data to make adjustments
adjustment1.fit(X, Y)
adjustment2.fit(X, Y)


# Variables assigned to the least squares (R^2) values ​​resulting from the fits
R_sq1 = adjustment1.score(X, Y)
R_sq2 = adjustment2.score(X, Y)

b0_adjustment1 = adjustment1.intercept_ # parameter b0 of adjustment1 (Linear Coefficient of the line, in this case we made b0 = 0)
b1_adjustment1 = adjustment1.coef_ # parameter b1 of adjustment1 (Angular Coefficient of the line)

b0_adjustment2 = adjustment2.intercept_ # parameter b0 of adjustment2 (Linear Coefficient of the line)
b1_adjustment2 = adjustment2.coef_ # adjustment2 parameter b (Angular Coefficient of the line)

print("\n========================================================")
print(" ** PARAMETERS OBTAINED FOR ADJUSTMENT 1 **")
print(" ========================================================")
print("\n Linear Coefficient for adjustment1 (b0): ", b0_adjustment1)
print("\n Angular Coefficient for adjustment1 (b1): ", b1_adjustment1)
print("\n R^2 value for adjustment1: ", R_sq1)

print("\n========================================================")
print(" ** PARAMETERS OBTAINED FOR ADJUSTMENT 2 **")
print(" ========================================================")
print("\n Linear Coefficient for adjustment2 (b0): ", b0_adjustment2)
print("\n Angular Coefficient for adjustment2 (b1): ", b1_adjustment2)
print("\n R^2 value for adjustment2: ", R_sq2)

# Viewing the graph
fig_size = plt.figure(figsize=(10,6))

plot_adjustment1 = fig_size.add_subplot(121)
plot_adjustment1 = plt.scatter(X, Y, color = 'blue')
plot_adjustment1 = plt.plot(X, adjustment1.predict(X), color = 'red')
plot_adjustment1 = plt.title('Ajuste 1')
plot_adjustment1 = plt.xlabel('X')
plot_adjustment1 = plt.ylabel('Y')

plot_adjustment2 = fig_size.add_subplot(122)
plot_adjustment2 = plt.scatter(X, Y, color = 'blue')
plot_adjustment2 = plt.plot(X, adjustment2.predict(X), color = 'red')
plot_adjustment2 = plt.title('Ajuste 2')
plot_adjustment2 = plt.xlabel('X')
plot_adjustment2 = plt.ylabel('Y')


plt.show()

