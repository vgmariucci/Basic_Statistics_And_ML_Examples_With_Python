################################################################################################

# We can predict a car's CO2 emissions based on its engine volume, but with multiple regression 
# we can include more variables, such as the car's weight, to make the prediction more accurate. 
# Considering the .csv file database, which contains information about some car makes and models, 
# develop a Python script to perform a linear regression of multiple variables and check the 
# relationship between CO2 emissions and car weight and engine volume.

################################################################################################

# Importing the libraries
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from matplotlib import pyplot as plt
from pathlib import Path

root_path = Path(__file__).parent
csv_file_name = 'CO2&CARS.csv'
csv_file_path = root_path / 'csv_files' / csv_file_name

# Defining the database
df = pd.read_csv(csv_file_path, sep = ';')

# Generating a list of predictor variables (independent variables) and naming them X [x1, x2, ..., xn]
X = df[['Volume_Engine', 'Car_Weight']]  # It is common to name the list of predictor variables with capital letters

# Defining the response variable (dependent variable)
y = df[['CO2']]    # It is common to name the list of response variables with lowercase letters.               

print("\n===============================================================================")
print("\n** GRAPHICAL REPRESENTATION OF LINEAR REGRESSION WITH 2 INDEPENDENT VARIABLES  **")
print("\n================================================================================")

fit_model = smf.ols(formula = 'CO2 ~ Volume_Engine + Car_Weight', data = df)
adjusted_equation = fit_model.fit()
adjusted_equation.params

# Preparing data for visualization
x_surf, y_surf = np.meshgrid(np.linspace(df.Volume_Engine.min(), df.Volume_Engine.max(), 100), np.linspace(df.Car_Weight.min(), df.Car_Weight.max(), 100))
onlyX = pd.DataFrame({'Volume_Engine': x_surf.ravel(), 'Car_Weight': y_surf.ravel()})
fittedY = adjusted_equation.predict(exog = onlyX)

# Convert the predicted results into an array
fittedY = np.array(fittedY)

# Graph construction for linear regression with multiple variables
fig = plt.figure(dpi = 100)
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(df['Volume_Engine'], df['Car_Weight'], df['CO2'], c = 'red', marker='o', alpha=0.5)
ax.plot_surface(x_surf, y_surf, fittedY.reshape(x_surf.shape), color='blue', alpha=0.3)
ax.set_xlabel('Volume_Engine (cm^3)')
ax.set_ylabel('Car_Weight (kg)')
ax.set_zlabel('CO2 (g/kg)')
plt.show()



# Add a constant to each predictor variable
X = sm.add_constant(X)

# Perform linear regression using the OLS() function --> Ordinary Least Squares
model = sm.OLS(y, X).fit()

# Shows the results of linear regression
print(model.summary())