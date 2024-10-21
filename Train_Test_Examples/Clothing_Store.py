########################################################################################################
#
# Suppose a clothing store wants to create a profit prediction model.
# To do so, the store intends to use a database with information about the shopping habits of 100 registered customers. The customer habits data were recorded by cameras located throughout the store. Whenever a customer approaches a certain clothing display, the cameras trigger a timer that counts the time between the customer choosing a particular item of clothing and paying at the checkout.
# You have been hired to develop a model using machine learning.
# To develop the model, the store provided you with the following information:
#
# Time interval in minutes that a customer takes to choose a product and pay at the checkout (predictor data or inputs);
# Value of the purchase made by the customer (response data or outputs).
#
###########################################################################################################

# Importing the libraries
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
np.random.seed(2)

# Predictor variable with the time intervals that
# the customer takes between choosing a product and paying at the checkout
# randomly generated with 100 samples distributed
# normally (Gaussian) around the mean of 3 minutes with
# standard deviation equal to +1 minute and -1 minute
time = np.random.normal(3, 1, 100) 

# Purchase values ​​also generated randomly
# with 100 normally distributed samples (Gaussian)
# around the mean of 150 dollars and standard deviation of + 40 dollars
# and - 40 dollars
purchase_value = np.random.normal(150, 40, 100)

################################################################################################
#
#                            EXPLORATORY PHASE (TIME TO INVESTIGATE!)
#
################################################################################################

plt.scatter(time, purchase_value)
plt.title('Time Purchase Value (Original Data)')
plt.xlabel('Time')
plt.ylabel('Purchase value')
plt.show()


# After checking the scatter plot of purchase values ​​as a function of the time interval
# spent by customers, you notice that everything looks very confusing and there is no relationship between the data
# or way to model the system using machine learning!

# So you think for a moment....And?

# Propõe a seguinte ideia na forma de pergunta: 
# 
# What if instead of trying to relate the purchase value directly to the time the customer spent
# (between choosing and paying for the product) we transform the response variable
# (output or dependent variable) by dividing it by the predictor variable?
# In other words, taking as the new response variable the ratio between the purchase value and
# the time spent by the customer to choose and pay for the product?

# Thus, the new response variable (output) is given by:

purchase_value_per_time =  purchase_value / time

# Generating the scatter plot of the new response variable as a function of time

plt.scatter(time, purchase_value_per_time)
plt.title('(Purchase Value / Time) Vs. Time')
plt.xlabel('Time')
plt.ylabel('Purchase Value / Time')
plt.show()

# From the new graph it is possible to notice a certain relationship between the new response variable
# and the predictor variable, so we can try to adjust some function using the machine learning method
# such as linear regression.

################################################################################################
#
#                      SEPARANDO OS DADOS PARA TREINOS E TESTES
#
################################################################################################

# Then you randomly split 70% of the original data to train the model
# and 30% of the data to test and check the accuracy of the model.

training_data_predictor = time[1:70]
training_data_response = purchase_value_per_time[1:70]

predictor_test_data = time[70:100] 
response_test_data = purchase_value_per_time[70:100]

# Displaying randomly separated data for training
plt.scatter(training_data_predictor, training_data_response)
plt.title('Training Data')
plt.xlabel('Training Data Predictor')
plt.ylabel('Training Data Response')
plt.show()

# Displaying randomly separated data for testing
plt.scatter(predictor_test_data, response_test_data)
plt.title('Test Data')
plt.xlabel('Predictor Test Data')
plt.ylabel('Test Data Response')
plt.show()

################################################################################################
#
#                                       TRAINING PHASE
#
################################################################################################


# Starting the linear regression process...
# We can ask the following question at this point in the modeling:

# What does the data set analyzed so far look like?

# One guess is to try to fit a polynomial function, that is, a polynomial linear regression!

# To perform this polynomial linear regression we can use the numpy library resource:

model_1 = np.poly1d(np.polyfit(training_data_predictor, training_data_response, 1))
curve_1 = np.linspace(0, 6, 100)
plt.scatter(training_data_predictor, training_data_response)
plt.plot(curve_1, model_1(curve_1), color='red')
plt.title('model 1 - > Adjustment with a straight line: y(x) = b1*x + b0')
plt.xlabel('training_data_predictor')
plt.ylabel('training_data_response')
plt.show()

model_2 = np.poly1d(np.polyfit(training_data_predictor, training_data_response, 2))
curve_2 = np.linspace(0, 6, 100)
plt.scatter(training_data_predictor, training_data_response)
plt.plot(curve_2, model_2(curve_2), color='red')
plt.title('model 2 - > Fit with a parabola: y(x) = b2*x^2 + b1*x + b0')
plt.xlabel('training_data_predictor')
plt.ylabel('training_data_response')
plt.show()

model_3 = np.poly1d(np.polyfit(training_data_predictor, training_data_response, 3))
curve_3 = np.linspace(0, 6, 100)
plt.scatter(training_data_predictor, training_data_response)
plt.plot(curve_3, model_3(curve_3), color='red')
plt.title('model 3 - > Fit with a polynomial of degree 3: y(x) = b3*x^3 + b2*x^2 + b1*x + b0')
plt.xlabel('training_data_predictor')
plt.ylabel('training_data_response')
plt.show()

model_4 = np.poly1d(np.polyfit(training_data_predictor, training_data_response, 4))
curve_4 = np.linspace(0, 6, 100)
plt.scatter(training_data_predictor, training_data_response)
plt.plot(curve_4, model_4(curve_4), color='red')
plt.title('model 4 - > Fit with a polynomial of degree 4: y(x) = b4*x^4 + b3*x^3 + b2*x^2 + b1*x + b0')
plt.xlabel('training_data_predictor')
plt.ylabel('training_data_response')
plt.show()

model_5 = np.poly1d(np.polyfit(training_data_predictor, training_data_response, 5))
curve_5 = np.linspace(0, 6, 100)
plt.scatter(training_data_predictor, training_data_response)
plt.plot(curve_5, model_5(curve_5), color='red')
plt.title('model 5 - > Fit with a polynomial of degree 5: y(x) = b5*x^5 + b4*x^4 + b3*x^3 + b2*x^2 + b1*x + b0')
plt.xlabel('training_data_predictor')
plt.ylabel('training_data_response')
plt.show()

# After analyzing each adjustment, we can see that the higher the degree of the polynomial, the better we get a curvee that describes the behavior of the training data. It is worth remembering that it is always possible to achieve the best adjustment with different polynomials. However, it is necessary to check whether the modeling process tends towards the case of overfitting (when the model is over-adjusted), in which it ceases to be accurate when we use the test data, presenting high variance. We can use the quality parameter R^2 of each adjustment to check which model is the best:

print("\n===============================================================================")
print("\n           ** R^2 OBTAINED FOR THE TRAINING PHASE  **"                            )
print("\n================================================================================")

r2_model_1 = r2_score(training_data_response, model_1(training_data_predictor))
print('\n R^2 model 1: ', r2_model_1)

r2_model_2 = r2_score(training_data_response, model_2(training_data_predictor))
print('\n R^2 model 2: ', r2_model_2)

r2_model_3 = r2_score(training_data_response, model_3(training_data_predictor))
print('\n R^2 model 3: ', r2_model_3)

r2_model_4 = r2_score(training_data_response, model_4(training_data_predictor))
print('\n R^2 model 4: ', r2_model_4)

r2_model_5 = r2_score(training_data_response, model_5(training_data_predictor))
print('\n R^2 model 5: ', r2_model_5)

################################################################################################
#
#                                      TESTING PHASE
#
################################################################################################

# After training the models, we can check which one will present the best result for the data randomly separated for the testing phase. To do this, we can use the R^2 parameter again:

model_1 = np.poly1d(np.polyfit(predictor_test_data, response_test_data, 1))
curve_1 = np.linspace(0, 6, 100)
plt.scatter(predictor_test_data, response_test_data)
plt.plot(curve_1, model_1(curve_1), color='red')
plt.title('model 1 - > Adjustment with a straight line: y(x) = b1*x + b0')
plt.xlabel('predictor_test_data')
plt.ylabel('response_test_data')
plt.show()

model_2 = np.poly1d(np.polyfit(predictor_test_data, response_test_data, 2))
curve_2 = np.linspace(0, 6, 100)
plt.scatter(predictor_test_data, response_test_data)
plt.plot(curve_2, model_2(curve_2), color='red')
plt.title('model 2 - > Fit with a parabola: y(x) = b2*x^2 + b1*x + b0')
plt.xlabel('predictor_test_data')
plt.ylabel('response_test_data')
plt.show()

model_3 = np.poly1d(np.polyfit(predictor_test_data, response_test_data, 3))
curve_3 = np.linspace(0, 6, 100)
plt.scatter(predictor_test_data, response_test_data)
plt.plot(curve_3, model_3(curve_3), color='red')
plt.title('model 3 - > Fit with a polynomial of degree 3: y(x) = b3*x^3 + b2*x^2 + b1*x + b0')
plt.xlabel('predictor_test_data')
plt.ylabel('response_test_data')
plt.show()

model_4 = np.poly1d(np.polyfit(predictor_test_data, response_test_data, 4))
curve_4 = np.linspace(0, 6, 100)
plt.scatter(predictor_test_data, response_test_data)
plt.plot(curve_4, model_4(curve_4), color='red')
plt.title('model 4 - > Fit with a polynomial of degree 4: y(x) = b4*x^4 + b3*x^3 + b2*x^2 + b1*x + b0')
plt.xlabel('predictor_test_data')
plt.ylabel('response_test_data')
plt.show()

model_5 = np.poly1d(np.polyfit(predictor_test_data, response_test_data, 5))
curve_5 = np.linspace(0, 6, 100)
plt.scatter(predictor_test_data, response_test_data)
plt.plot(curve_5, model_5(curve_5), color='red')
plt.title('model 5 - > Fit with a polynomial of degree 5: y(x) = b5*x^5 + b4*x^4 + b3*x^3 + b2*x^2 + b1*x + b0')
plt.xlabel('predictor_test_data')
plt.ylabel('response_test_data')
plt.show()

print("\n===============================================================================")
print("\n           ** R^2 OBTAINED FOR THE TESTING PHASE  **"                            )
print("\n================================================================================")

r2_model_1 = r2_score(response_test_data, model_1(predictor_test_data))
print('\n R^2 model 1: ', r2_model_1)

r2_model_2 = r2_score(response_test_data, model_2(predictor_test_data))
print('\n R^2 model 2: ', r2_model_2)

r2_model_3 = r2_score(response_test_data, model_3(predictor_test_data))
print('\n R^2 model 3: ', r2_model_3)

r2_model_4 = r2_score(response_test_data, model_4(predictor_test_data))
print('\n R^2 model 4: ', r2_model_4)

r2_model_5 = r2_score(response_test_data, model_5(predictor_test_data))
print('\n R^2 model 5: ', r2_model_5)


################################################################################################
#
#                                   PREDICTING PROFITS
#
################################################################################################

# Assuming that we choose model_4 to predict the purchase value of each customer based on the time it takes to choose one or more products and make the purchase, we can test some new values ​​that could be of interest to the clothing store: How much would a customer spend in the store if the time between choosing a certain product and paying at the checkout is equal to 5 minutes?


model_4 = np.poly1d(np.polyfit(time, purchase_value_per_time, 4))
curve_4 = np.linspace(0, 6, 100)
plt.scatter(time, purchase_value_per_time)
plt.plot(curve_4, model_4(curve_4), color='red')
plt.title('model 4 - > Model Chosen to Predict Store Profits')
plt.xlabel('time (min)')
plt.ylabel('(Purchase Value / time) (U$/min)')
plt.show()


print("\n Purchase value predicted by model 4 for the 5-minute time interval: ", model_4(5))
