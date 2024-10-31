#############################################################################################################
#
# Suppose we want to know whether the number of hours spent studying and the number of practice tests had
# an effect on the grade a given student gets on an official test. To explore this relationship,
# we can apply the multivariate linear regression method in Python.
#
#############################################################################################################
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from matplotlib import pyplot as plt

df = pd.DataFrame({'study_hours': [1, 2, 2, 4, 2, 1, 5, 4, 2, 4, 4, 3, 6, 5, 3, 4, 6, 2, 1, 2],
                   'number_of_simulations': [1, 3, 3, 5, 2, 2, 1, 1, 0, 3, 4, 3, 2, 4, 4, 4, 5, 1, 0, 1],
                   'test_grades': [76, 78, 85, 88, 72, 69, 94, 94, 88, 92, 90, 75, 96, 90, 82, 85, 99, 83, 62, 76]})


print("\n===============================================================================")
print("\n** GRAPHICAL REPRESENTATION OF LINEAR REGRESSION WITH 2 INDEPENDENT VARIABLES  **")
print("\n================================================================================")

fit_model = smf.ols(formula = 'test_grades ~ study_hours + number_of_simulations', data = df)
adjusted_equation = fit_model.fit()
adjusted_equation.params

# Preparing data for visualization
x_surf, y_surf = np.meshgrid(np.linspace(df.study_hours.min(), df.study_hours.max(), 100), np.linspace(df.number_of_simulations.min(), df.number_of_simulations.max(), 100))
onlyX = pd.DataFrame({'study_hours': x_surf.ravel(), 'number_of_simulations': y_surf.ravel()})
fittedY = adjusted_equation.predict(exog = onlyX)

# Convert the predicted results into an array
fittedY = np.array(fittedY)

# Building a graph for linear regression with multiple variables
fig = plt.figure(dpi = 100)
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(df['study_hours'], df['number_of_simulations'], df['test_grades'], c = 'red', marker='o', alpha=0.5)
ax.plot_surface(x_surf, y_surf, fittedY.reshape(x_surf.shape), color='blue', alpha=0.3)
ax.set_xlabel('study_hours')
ax.set_ylabel('number_of_simulations')
ax.set_zlabel('test_grades')
plt.show()


# Define the response variable (dependent variable)
y = df['test_grades']

# Define the predictor variables (independent variables)
x = df[['study_hours', 'number_of_simulations']]

# Add a constant to each predictor variable
x = sm.add_constant(x)

# Perform linear regression using the OLS() function --> Ordinary Least Squares
modelo = sm.OLS(y, x).fit()

# Shows the results of linear regression
print(modelo.summary())

################################################################
#
# Below we have the interpretation for some of the parameters
# resulting from linear regression
#
################################################################

# R-squared: 0.734

# This is the result of least squares and is known as the
# coefficient of determination. It represents the proportion of the variance
# for the response variable (dependent variable) that can be explained
# by the predictor variables (independent variables). In this example,
# 73.4% of the variance in official test scores can be explained
# by the number of hours studied and the number of practice tests.
#
###################################################################

# F-statistic: 23.46

# The overall significance F-test indicates whether your linear regression model
# provides a better fit to the data than a model that contains no
# independent variables
#
####################################################################

# Prob (F-statistic): 1.29e-05

# This is the p-value associated with the overall F-statistic. 
# It tells us whether the regression model as a whole is statistically significant. 
# In other words, it tells us whether the two predictor variables combined have a statistically 
# significant association with the response variable. In this case, the p-value is less than 0.05, 
# which indicates that the predictor variables “study hours” and “number of mock exams” combined 
# have a statistically significant association with official test scores.

#coef: 

# The coefficients for each predictor variable tell us the expected average change in the response variable, 
# assuming the other predictor variable remains constant. 
# # For example, for each additional hour of studying, the average exam score is expected to increase by 5.56, 
# assuming the preparatory exams taken remain constant.

# Here's another way to think about it:

# if Student A and Student B take the same number of practice tests,
# but Student A studies for an extra hour, Student A would be expected to
# score 5.56 points higher than Student B.

# We interpret the coefficient for the intercept as meaning that the expected grade for a student who studies zero hours 
# and takes zero preparatory exams is 67.67.

# P>|t|:

# The individual p-values ​​tell us whether each predictor variable is statistically
# significant. We can see that “hours of study” is statistically significant (p = 0.00)
# while “number of mock exams” (p = 0.52) is not statistically significant.
# Since “number of mock exams” is not statistically significant, we may
# decide to remove it from the model.

# Estimated regression equation:
# 
# We can use the coefficients from the model output to create the following estimated regression equation:

# official test scores = 67.67 + 5.56*(hours of study) – 0.60*(number of mock tests)
