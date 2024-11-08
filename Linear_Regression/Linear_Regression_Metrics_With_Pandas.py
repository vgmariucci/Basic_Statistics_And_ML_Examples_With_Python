################################################################################################
#
# Example of how to perform simple linear regression, with emphasis on comparing
# the least squares R^2 performance metric or parameter of each fit
# 
################################################################################################

# Importing the libraries
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


# Creating the database
dados = {'X': [5, 15, 25, 35, 45, 55],
         'Y': [5, 20, 14, 32, 22, 38]}

dados = pd.DataFrame(data = dados)

# Separating the data:
# X is the independent variable
# Y is the dependent variable
X = dados['X'].values
Y = dados['Y'].values

# Function to transpose the X coordinate values
X = X.reshape(-1, 1)

# Creating the instances for the linear regression classes
fit1  = LinearRegression(fit_intercept= False) # y(x) = b1*x  (b0 = 0)
fit2  = LinearRegression(fit_intercept= True)  # y(x) = b0 + b1*x


# Passing the data to make the adjustments
fit1.fit(X, Y)
fit2.fit(X, Y)


# Variables assigned to the least squares values ​​(R^2) resulting from the fits
R_sq1 = fit1.score(X, Y)
R_sq2 = fit2.score(X, Y)

b0_fit1 = fit1.intercept_ # parameter b0 of fit 1 (Linear Coefficient of the line, in this case we made b0 = 0)
b1_fit1 = fit1.coef_ # parameter b1 of fit1 (Angular Coefficient of the line)

b0_fit2 = fit2.intercept_ # parameter b0 of fit2 (Linear Coefficient of the line)
b1_fit2 = fit2.coef_ # parameter b of fit2 (Angular Coefficient of the line)

print("\n========================================================")
print(" ** PARAMETERS OBTAINED FOR FIT 1 **")
print(" ========================================================")
print("\n Linear Coefficient for fit1 (b0): ", b0_fit1)
print("\n Angular Coefficient for fit1 (b1): ", b1_fit1)
print("\n R^2 value for fit1: ", R_sq1)

print("\n========================================================")
print(" ** PARAMETERS OBTAINED FOR FIT 2 **")
print(" ========================================================")
print("\n Linear Coefficient for fit1 (b0): ", b0_fit2)
print("\n Angular Coefficient for fit1 (b1): ", b1_fit2)
print("\n R^2 value for fit1: ", R_sq2)


# Viewing the graph
fig_size = plt.figure(figsize=(10,6))

plot_fit1 = fig_size.add_subplot(121)
plot_fit1 = plt.scatter(X, Y, color = 'blue')
plot_fit1 = plt.plot(X, fit1.predict(X), color = 'red')
plot_fit1 = plt.title('fit 1')
plot_fit1 = plt.xlabel('X')
plot_fit1 = plt.ylabel('Y')

plot_fit2 = fig_size.add_subplot(122)
plot_fit2 = plt.scatter(X, Y, color = 'blue')
plot_fit2 = plt.plot(X, fit2.predict(X), color = 'red')
plot_fit2 = plt.title('fit 2')
plot_fit2 = plt.xlabel('X')
plot_fit2 = plt.ylabel('Y')


plt.show()

