#############################################################################################################
#
# Example of how to perform multiple linear regression considering 2 independent variables,
# with emphasis on comparing the least squares R^2 performance metric of each fit
# 
##############################################################################################################

# Importing the libraries
import numpy as np
from sklearn.linear_model import LinearRegression

import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf


# Example data
x= [[2, 1], [5, 3], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35],
    [64, 40], [70, 41], [78, 32], [83, 45], [90, 51], [95, 55], [100, 64], [104, 65]]
y= [4, 5, 20, 14, 32, 42, 38, 43, 36, 25, 50, 44, 52, 62, 68, 73]


# Creating the instance for the linear regression classes
adjust1  = LinearRegression(fit_intercept= False) # y(x1, x2) = b1*x1 + b2*x2  (b0 = 0)
adjust2  = LinearRegression(fit_intercept= True)  # y(x1, x2) = b0 + b1*x + b2*x2


# Passing the data to make the adjustments
adjust1.fit(x, y)
adjust2.fit(x, y)

# Variables assigned to the least squares values ​​(R^2) resulting from the adjustments
R_sq1 = adjust1.score(x, y)
R_sq2 = adjust2.score(x, y)

b0_adjust1 = adjust1.intercept_ # parameter b0 of adjust1 (Linear Coefficient of the line, in this case we made b0 = 0)
b1_adjust1 = adjust1.coef_[0] # parameter b1 of adjust1 (Angular Coefficient of the line when b2 = 0)
b2_adjust1 = adjust1.coef_[1] # parameter b2 of adjust1 (Angular Coefficient of the line when b1 = 0)

b0_adjust2 = adjust2.intercept_ # adjust2 parameter b0 (Linear Coefficient of the line)
b1_adjust2 = adjust2.coef_[0] # parameter b1 of adjust2 (Angular Coefficient of the line when b2 = 0)
b2_adjust2 = adjust2.coef_[1] # b2 parameter of adjust2 (Angular Coefficient of the line when b1 = 0)

print("\n========================================================")
print("\n** PARAMETERS OBTAINED FOR adjust 1 **")
print("\n========================================================")
print("\n Linear Coefficient for adjust1 (b0): ", b0_adjust1)
print("\n Angular Coefficient b1 for adjust1: ", b1_adjust1)
print("\n Angular Coefficient b2 for adjust1: ", b2_adjust1)
print("\n R^2 value for adjust1: ", R_sq1)

print("\n========================================================")
print("\n** PARAMETERS OBTAINED FOR adjust 2 **")
print("\n========================================================")
print("\n Linear Coefficient for (b0): ", b0_adjust2)
print("\n Angular Coefficient b1 for adjust2: ", b1_adjust2)
print("\n Angular Coefficient b2 for adjust2: ", b2_adjust2)
print("\n R^2 value for adjust2: ", R_sq2)


print("\n===============================================================================")
print("\n** GRAPHICAL REPRESENTATION OF LINEAR REGRESSION WITH 2 INDEPENDENT VARIABLES  **")
print("\n================================================================================")
# Multiple linear regression using pandas and statsmodels
df = pd.DataFrame(x, columns=['x1', 'x2'])
df['y'] = pd.Series(y)

adjust_model = smf.ols(formula = 'y ~ x1 + x2', data = df)
adjusted_equation = adjust_model.fit()
adjusted_equation.params

# Preparing data for visualization	
x_surf, y_surf = np.meshgrid(np.linspace(df.x1.min(), df.x1.max(), 100), np.linspace(df.x2.min(), df.x2.max(), 100))
onlyX = pd.DataFrame({'x1': x_surf.ravel(), 'x2': y_surf.ravel()})
fittedY = adjusted_equation.predict(exog = onlyX)

# Convert the predicted results into an array
fittedY = np.array(fittedY)

# Construction of the graph for linear regression with multiple variables
fig = plt.figure(dpi = 100)
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(df['x1'], df['x2'], df['y'], c = 'red', marker='o', alpha=0.5)
ax.plot_surface(x_surf, y_surf, fittedY.reshape(x_surf.shape), color='blue', alpha=0.3)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.show()
