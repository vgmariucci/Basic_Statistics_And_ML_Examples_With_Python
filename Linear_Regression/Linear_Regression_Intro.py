########################################################################
#
#   Example of how to perform linear regression
#
########################################################################

# Importing the libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

# Creating the database
dados = {'Health Plan Value': [200, 220, 300, 290, 450, 457, 500, 530, 700, 800],
         'Age': [18,   22,  23,  30,  35,  44,  49,  50,  67,  75]}

dados = pd.DataFrame(data = dados)

# Separating the data:
# X is the independent variable
# Y is the dependent variable
X = dados['Age'].values
Y = dados['Health Plan Value'].values

# Function to use transposed X
X = X.reshape(-1,1)

# Defining the linear regressor
regressor = LinearRegression()

# Passing the data to train the regressor
regressor.fit(X,Y)

# Viewing the graph
plt.scatter(X,Y, color = 'black')
plt.plot(X, regressor.predict(X), color = 'red')
plt.title('Simple Linear Regression')
plt.xlabel('Age')
plt.ylabel('Health Plan Value')
plt.show()

# Predicting new values
age = np.array(57)
# Two ways to pass values ​​to the function that has been adjusted
prediction_1 = regressor.predict(age.reshape(-1, 1))
prediction_2 = regressor.intercept_ + regressor.coef_*age # Pass the age value to the fitted line function.
                                                         # In this case represented in explicit form: f(x) = b0 + b1.x
print(f"For the age {age}, the predicted Health Plan Values were:\n")
print("\n Previsão 1: ", prediction_1)
print("\n Previsão 2: ", prediction_2)

