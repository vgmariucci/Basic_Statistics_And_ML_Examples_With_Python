################################################################################################
#
# Example of how to perform polynomial linear regression, with emphasis on comparing
# the least squares R^2 performance metric for each fit
# 
################################################################################################

# Importing useful libraries into this script
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Database
data={'X': [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22],
'Y': [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]}

data = pd.DataFrame(data = data)

# Separating the data:
# X is the independent variable
# Y is the dependent variable
X = data['X'].values
Y = data['Y'].values

fit_degree_1 = np.poly1d(np.polyfit(X, Y, 1))  # y(x) = b0 + b1*x (Line Equation)
fit_degree_1_line = np.linspace(1, 22, 100)
# Calculates the R^2 value for the performed fit
R_sq1 = r2_score(Y, fit_degree_1(X))  


# Performing fits with polynomial functions of degree 2, 3, 4 and 5

fit_degree_2 = np.poly1d(np.polyfit(X, Y, 2))                      # Polynomial function of degree 2 (y(x) = b2*x^2 + b1*x + b0)
fit_degree_2_line = np.linspace(1, 22, 100)
# Calculates the R^2 value for the performed fit
R_sq2 = r2_score(Y, fit_degree_2(X))             

fit_degree_3 = np.poly1d(np.polyfit(X, Y, 3))                      # Polynomial function of degree 3 (y(x) = b3*x^3 + b2*x^2 + b1*x + b0)
fit_degree_3_line = np.linspace(1, 22, 100)
# Calculates the R^2 value for the performed fit
R_sq3 = r2_score(Y, fit_degree_3(X))             

fit_degree_4 = np.poly1d(np.polyfit(X, Y, 4))                      # Polynomial function of degree 4 ( y(x) = b4*x^4 + b3*x^3 + b2*x^2 + b1*x + b0 )
fit_degree_4_line = np.linspace(1, 22, 100)
# Calculates the R^2 value for the performed fit
R_sq4 = r2_score(Y, fit_degree_4(X))             

fit_degree_5 = np.poly1d(np.polyfit(X, Y, 5))                      # Polynomial function of degree 5 ( y(x) = b5*x^5 + b4*x^4 + b3*x^3 + b2*x^2 + b1*x + b0 )
fit_degree_5_line = np.linspace(1, 22, 100)
# Calculates the R^2 value for the performed fit
R_sq5 = r2_score(Y, fit_degree_5(X))             



print("\n========================================================")
print("\n** PARAMETERS OBTAINED FOR THE LINE **")
print("\n========================================================")
print("\n Value of R^2 for the fit polynomial of degree 1: ", R_sq1)

plot_fit_degree_1 = plt.scatter(X, Y, color = 'blue')
plot_fit_degree_1 = plt.plot(fit_degree_1_line, fit_degree_1(fit_degree_1_line), color = 'red')
plot_fit_degree_1 = plt.title('fit with a first-degree function: y(x) = b0 + b1*x')
plot_fit_degree_1 = plt.xlabel('X')
plot_fit_degree_1 = plt.ylabel('Y')
plt.show()



print("\n========================================================")
print("\n** PARAMETERS OBTAINED FOR THE DEGREE POLYNOMIAL FIT 2 **")
print("\n========================================================")
print("\n Value of R^2 for the fit polynomial of degree 2: ", R_sq2)

plot_fit_degree_2 = plt.scatter(X, Y, color = 'blue')
plot_fit_degree_2 = plt.plot(fit_degree_2_line, fit_degree_2(fit_degree_2_line), color = 'red')
plot_fit_degree_2 = plt.title('fit with a quadratic function: y(x) = b2*x^2 + b1*x + b0')
plot_fit_degree_2 = plt.xlabel('X')
plot_fit_degree_2 = plt.ylabel('Y')
plt.show()

print("\n========================================================")
print("\n** PARAMETERS OBTAINED FOR THE DEGREE POLYNOMIAL FIT 3 **")
print("\n========================================================")
print("\n Value of R^2 for the fit polynomial of degree 3: ", R_sq3)

plot_fit_degree_3 = plt.scatter(X, Y, color = 'blue')
plot_fit_degree_3 = plt.plot(fit_degree_3_line, fit_degree_3(fit_degree_3_line), color = 'red')
plot_fit_degree_3 = plt.title('fit with a third-degree function: y(x) = b3*x^3 + b2*x^2 + b1*x + b0')
plot_fit_degree_3 = plt.xlabel('X')
plot_fit_degree_3 = plt.ylabel('Y')
plt.show()


print("\n========================================================")
print("\n** PARAMETERS OBTAINED FOR THE DEGREE POLYNOMIAL FIT 4 **")
print("\n========================================================")
print("\n Value of R^2 for the fit polynomial of degree 4: ", R_sq4)


plot_fit_degree_4 = plt.scatter(X, Y, color = 'blue')
plot_fit_degree_4 = plt.plot(fit_degree_4_line, fit_degree_4(fit_degree_4_line), color = 'red')
plot_fit_degree_4 = plt.title('fit with a fourth-degree function: y(x) = b4*x^4 + b3*x^3 + b2*x^2 + b1*x + b0')
plot_fit_degree_4 = plt.xlabel('X')
plot_fit_degree_4 = plt.ylabel('Y')
plt.show()


print("\n========================================================")
print("\n** PARAMETERS OBTAINED FOR THE DEGREE POLYNOMIAL FIT 5 **")
print("\n========================================================")
print("\n Value of R^2 for the fit polynomial of degree 5: ", R_sq5)

plot_fit_degree_5 = plt.scatter(X, Y, color = 'blue')
plot_fit_degree_5 = plt.plot(fit_degree_5_line, fit_degree_5(fit_degree_5_line), color = 'red')
plot_fit_degree_5 = plt.title('fit with a fifth-degree function: y(x) = b5*x^5 + b4*x^4 + b3*x^3 + b2*x^2 + b1*x + b0')
plot_fit_degree_5 = plt.xlabel('X')
plot_fit_degree_5 = plt.ylabel('Y')
plt.show()