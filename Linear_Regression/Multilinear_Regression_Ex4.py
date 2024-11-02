################################################################################################
# We can predict a car's CO2 emissions based on its engine volume, but with multiple regression 
# we can include more variables, such as the car's weight, to make the prediction more accurate. 
# Considering the .csv file database, which gathers information about some car brands and models, 
# develop a Python script to perform a linear regression of multiple variables and verify the 
# relationship between CO2 emissions with the car's weight and engine volume. 

# In this example we will analyze some important statistical properties, which are usually 
# verified during the construction of a machine learning model.

################################################################################################

# Importing the base libraries
import pandas as pd
import numpy as np


# Libraries used for building a machine learning model
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from statsmodels.stats.diagnostic import normal_ad
from statsmodels.stats.stattools import durbin_watson

# Libraries for building graphs
import seaborn as sn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

root_path = Path(__file__).parent
csv_file_name = 'CO2&CARS.csv'
csv_file_path = root_path / 'csv_files' / csv_file_name

# Defining the database
df = pd.read_csv(csv_file_path, sep = ';')

plt.rcParams['figure.figsize'] = (7,7)
plt.style.use('ggplot')

print(df)

########################################################################
#
#    EXPLORING THE EXISTENCE OF SOME RELATIONSHIP BETWEEN THE DATA
#
########################################################################

# Visualizing data using scatter plots and histograms
sn.set_palette('colorblind')
sn.pairplot(data = df, height = 3)
plt.show()

# We can see from the graphs generated with the seaborn library function that 
# there is a positive trend relationship between the data in the Engine_Volume 
# and Car_Weight columns, as well as between the CO2 columns with Engine_Volume 
# and CO2 with Car_Weight. 

########################################################################
#
#                CONSTRUCTION OF THE LINEAR REGRESSION MODEL
#
########################################################################

# Generating a list of predictor variables (independent variables) and naming them X [x1, x2, ..., xn]
X = df[['Volume_Engine', 'Car_Weight']]  # It is common to name the list of predictor variables with capital letters

# Defining the response variable (dependent variable)
y = df[['CO2']]    # It is common to name the list of response variables with lowercase letters.             

ajuste_modelo = smf.ols(formula = 'CO2 ~ Volume_Engine + Car_Weight', data = df)
equacao_ajustada = ajuste_modelo.fit()
equacao_ajustada.params

# Adds a constant for each predictor variable
X = sm.add_constant(X)

# Perform linear regression using the OLS() function --> Ordinary Least Squares
model = sm.OLS(y, X).fit()

# Shows the results of linear regression
print(model.summary())

########################################################################
#
#                PREDICTING CO2 EMISSION VALUES WITH THE MODEL
#
########################################################################
def calculate_CO2_level(x1,x2):
        
    CO2 = 79.6947 + 0.0078 * x1 + 0.0076 * x2

    return CO2
    

try:
    Volume_Engine = int(input(" Enter the vehicle's engine volume: \n"))
    Car_Weight = int(input("Enter the weight of the vehicle: \n"))
    print("\nLevel of CO2 emitted by the vehicle: \n", calculate_CO2_level(Volume_Engine, Car_Weight))

except ValueError:
    print("\nPlease provide only the integer numeric values ​​of each quantity.: \n Ex: Engine Volume = 1200 \n Car Weight = 2000")


########################################################################
#
#                VISUALIZATION OF DATA AND ADJUSTED MODEL
#
########################################################################

# Preparing data for visualization
x_surf, y_surf = np.meshgrid(np.linspace(df.Volume_Engine.min(), df.Volume_Engine.max(), 100), np.linspace(df.Car_Weight.min(), df.Car_Weight.max(), 100))
onlyX = pd.DataFrame({'Volume_Engine': x_surf.ravel(), 'Car_Weight': y_surf.ravel()})
fittedY = equacao_ajustada.predict(exog = onlyX)

# Convert the predicted results into an array
fittedY = np.array(fittedY)

# Building a graph for linear regression with multiple variables
fig = plt.figure(dpi = 100)
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(df['Volume_Engine'], df['Car_Weight'], df['CO2'], c = 'red', marker='o', alpha=0.5)
ax.plot_surface(x_surf, y_surf, fittedY.reshape(x_surf.shape), color='blue', alpha=0.3)
ax.set_xlabel('Volume_Engine (cm^3)')
ax.set_ylabel('Car_Weight (kg)')
ax.set_zlabel('CO2 (g/kg)')
plt.show()


########################################################################
#
#                           MODEL VALIDATION
#
########################################################################
# After building the model, it is important for us to validate its performance.
# We can evaluate a model by looking at its coefficient of determination ( R2 ),
# F-test, t-test and also the residuals.
#
# The model summary contains many important values ​​that we can use
# to evaluate our model.

# The coefficient of determination R2 is the portion of the total variation of the
# dependent variable that is explained by the variation of the independent variable.

# With the statsmodel lib we can obtain the R2 value of our model by accessing the
# .rsquared attribute of our model.
print("\n Coefficient of determination R2: ", model.rsquared)

# R2 ranges from 0 to 1, 
# where R2=0 means there is no linear relationship between the variables
# and R2 = 1 shows a perfect linear relationship. 
# In our case, we got an R2 score of about 0.3765, which means that 37.65% of our dependent 
# variable can be explained using our independent variables.


# F-Test (ANOVA)
#
# The F-test or ANOVA (Analysis of Variance) in multilinear regression
# can be used to determine whether our complex model performs better than a
# simpler model (e.g., a model with only one independent variable).
# With the F-test we can assess the significance of our model by calculating the probability
# of observing an F-statistic that is at least as high as the value our model obtained.

# Similar to the R2 score, we can easily obtain the F-statistic and the probability of said
# F-statistic by accessing the .fvalue and .f_pvalue attribute of our model as follows:
print("\n F-statistic (ANOVA): ", model.fvalue)
print("\n Probability (f_pvalue) of observing values ​​greater than F-statistic: ", model.f_pvalue)

# Since our f_pvalue is less than 0.05, we can conclude that our model performs better
# than another simpler model (e.g. without considering the influence of independent variables).


# T-Test
#
# The t-statistic parameter is the linear coefficient divided by its standard error.
# The standard error is an estimate of the standard deviation of the coefficient, the amount that varies between cases.
# It can be thought of as a measure of how precisely the regression coefficient is measured.
# Like the F-test, the p-value shows the probability of seeing a result that is as extreme as our model.
# We can also get the p-value for all of our variables by calling the .pvalues ​​attribute on the model.
print(model.pvalues)

# Both independent variables, Volume_Engine and Car_Weight, have p_value greater than 0.05,
# this means that there is not enough evidence that Volume_Engine and Car_Weight affect
# CO2 emission levels.

########################################################################
#
#                       ASSUMPTION TESTING
#
########################################################################

# Next, we will validate our model by performing residual analysis,
# Below is the list of tests or assumptions that we will perform to verify the validity of our model:

# Linearity
# Normality
# Multicollinearity
# Autocorrelation
# Homoscedasticity

# Residual is the difference between the observed value and the predicted value of our dataset.
# With statsmodel we can easily get the residual value of our model by simply accessing
# the .resid attribute of the model and then we can keep it in a new column called 'residual'
# in our dataframe df.
df['Predicted_CO2_Level'] = model.predict(X)
df['Residual'] = model.resid
print(df) 


# Linearity
#
# Assumes that there is a linear relationship between the independent variables and the dependent variable.
# In our case, since we have 2 independent variables, we can do this using a scatterplot
# to see our predicted values ​​versus the actual values.

# Building the graph of actual values ​​and predicted values
sn.lmplot(x = 'CO2', y = 'Predicted_CO2_Level', data = df, fit_reg = False)

# Construction of the diagonal line
line_coordinates = np.arange(df[['CO2','Predicted_CO2_Level']].min().min()-10,
                              df[['CO2','Predicted_CO2_Level']].max().min()+10)

plt.plot(line_coordinates, line_coordinates, # points X e y
         color = 'darkorange', linestyle='--')

plt.ylabel('Predicted_CO2_Level', fontsize= 14)
plt.xlabel('Real Value of CO2', fontsize= 14)
plt.title('Linearity Assumption', fontsize=14)
plt.show()

# The scatter plots show residual points distributed unevenly enough around the diagonal line,
# so that we cannot assume that there is a linear relationship between our independent and dependent variables.
# This is an important assessment and easy to understand in this specific example, since the weight of the car is
# strongly influenced by the weight of the engine, and consequently by its size, i.e. the volume of the engine!


# Normality
#
# This assumes that the error terms of the model are normally distributed.
# We will test the normality of the residuals by plotting them on the histogram and looking at the p_value from the 
# Anderson-Darling test for normality. We will use the normal_ad() function from statsmodel to calculate our p_value
# and then compare it to the threshold of 0.05.
# If the p_value we get is greater than the threshold, we can assume that our residual is normally distributed.
p_value = normal_ad(df['Residual'])[1]
print("\n p_value obtained from the Anderson-Darling test:", p_value)
print("\n p_values ​​< 0.05 means a non-normal distribution")

# Plotting the distribution of residuals
plt.subplots(figsize = (8,4))
plt.title('Residual distribution', fontsize = 18)
sn.distplot(df['Residual'])
plt.show()

# Analysis of the normality of the residues
if p_value < 0.05:
    print("\n The residues are not normally distributed")
else:
    print("\n Residues is normally distributed")
    
# From the above code, we get a p_value = 0.1816, which can be considered normal because it is above the threshold of 0.05.
# The histogram plot also shows a normal distribution (although it looks a bit skewed because we have so few observations
# in our dataset). From both results, we can assume that our residuals are normally distributed.


# Multicollinearity
#
# This assumes that the predictors used in the regression are not correlated with each other.
# To identify if there is any correlation between our predictors, we can calculate the Pearson correlation coefficient
# between each column in our data using the Pandas dataframe's corr() function.
# We can then display it as a heatmap using Seaborn's heatmap() function.
corr = df[['Volume_Engine', 'Car_Weight', 'CO2']].corr()
print("\nMactress of Pearson's correlation coefficients for each variable:\n", corr)

# Generates a mask for the elements of the main diagonal of the matrix
diagonal_mask = np.zeros_like(corr, dtype = np.bool)
np.fill_diagonal(diagonal_mask, val = True)

# Defines the size of the figure for constructing the correlation matrix
fig, ax = plt.subplots(figsize = (4,3))

# Generate a custom colormap to differentiate extreme values ​​in a certain range
color_map = sn.diverging_palette(220, 10, as_cmap = True, sep = 100)
color_map.set_bad('grey')

# Construct the colormap with the correlation matrix mask and size defined above
sn.heatmap(corr, mask = diagonal_mask, cmap = color_map, vmin = -1, vmax = 1, center = 0, linewidths=.5)
fig.suptitle('Pearson correlation coefficient matrix', fontsize= 24)
ax.tick_params(axis='both', which='major', labelsize = 10)
plt.show()

# The matrix image shows that there is a strong positive relationship between Volume_Engine and Car_Weight
# and a weaker positive relationship between CO2 with Volume_Engine and CO2 with Car_Weight.
# This means that both independent variables are affecting each other and that there is
# multicollinearity in our data.


# Autocorrelation

# Autocorrelation is the correlation of errors (residuals) over time.
# Used when data is collected over time to detect if autocorrelation is present.
# Autocorrelation exists if the residuals in one time period are related to the residuals in another time period.
# We can detect autocorrelation by performing the Durbin-Watson test to determine if there is a positive or negative correlation.
# In this step, we will use the statsmodel durbin_watson() function to calculate our Durbin-Watson score and then
# evaluate the value with the following condition:

# If the Durbin-Watson score is less than 1.5, then there is positive autocorrelation and the assumption is not met
# If the Durbin-Watson score is between 1.5 and 2.5, then there is no autocorrelation and the assumption is met
# If the Durbin-Watson score is greater than 2.5, then there is negative autocorrelation and the assumption is not met
durbinWatson = durbin_watson(df['Residual'])

print('Durbin-Watson: ', durbinWatson)

if durbinWatson < 1.5:
    print("\n Positive autocorrelation signal,")
    print("\n Unsatisfied assumption")
elif durbinWatson > 2.5:
    print("\n Negative autocorrelation signal,")
    print("\n Unsatisfied assumption")
else:
    print("\n Low autocorrelation signal,")
    print("\n Assumption satisfied")

# Our model obtained a Durbin-Watson score of about 0.94, which is below 1.5,
# so we can assume that there is autocorrelation in our residual.

# Homoscedasticity
#
# This assumes homoscedasticity, which is the same variance within our error terms.
# Heteroscedasticity, the violation of homoscedasticity, occurs when we do not have a uniform
# variance across the error terms. To detect homoscedasticity, we can plot our residual
# and see if the variance appears to be uniform.

# Plotting the residuals
plt.subplots(figsize = (8,4))
plt.scatter(x = df.index, y = df.Residual, alpha=0.8)
plt.plot(np.repeat(0, len(df.index)+2), color = 'darkorange', linestyle = '--')

plt.ylabel('Residual', fontsize = 14)
plt.xlabel('Sample Number', fontsize = 14)
plt.title('Homoscedasticity Test', fontsize = 16)
plt.show()

# Despite having few data points, our residual appears to have constant and uniform variance,
# so we can assume that it satisfies the homoscedasticity assumption.

# Conclusion
#
# Our model did not pass all the tests in the model validation stages,
# however, we can conclude that it can perform well in predicting CO2 emission levels
# for the two independent variables, Engine Volume and Car Weight.
# But still, our model only has an R2 score of 37.65%,
# which means that there are still about 62% of unknown factors that are affecting CO2 emissions.
