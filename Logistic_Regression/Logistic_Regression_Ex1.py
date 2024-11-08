###########################################################################################################
#
# Suppose you want to create a classification model using logistic regression to predict
# whether you will pass or fail a college course.

# The data you have are:

# Hours of study per week;
# Study methods A and B;
# Results of previous exams: whether you passed or failed.
##########################################################################################################

# Importing the libraries
import pandas as pd
import statsmodels.formula.api as smf
from sklearn import metrics 
import seaborn as sn
import matplotlib.pyplot as plt


# Creating the training dataframe
df_training = pd.DataFrame({'test_results': [1, 1, 0, 1, 0, 1, 0, 1, 1, 0,
                              0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                   'study_hours': [2, 4, 5, 6, 2, 3, 2, 1, 8, 6,
                            5, 8, 8, 7, 6, 7, 5, 4, 8, 9],
                   'study_methods': ['A', 'A', 'A', 'B', 'B', 'B', 'B',
                             'B', 'B', 'A', 'B', 'A', 'B', 'B',
                             'A', 'A', 'B', 'A', 'B', 'A']})

# Print the dataframe just for checking and debugging if necessary
print(df_training)

############################################################################################
#
#                          TRAINING THE LOGISTIC REGRESSION MODEL
#
############################################################################################

# Using the logit() function from the statsmodels library we can perform logistic regression
model = smf.logit('test_results ~ study_hours + study_methods', data = df_training)
model = model.fit()

# We present the result after training the model with logistic regression.
print("\n==========================================================================================")
print("\n   ** RESULTS OBTAINED FOR TRAINING THE LOGISTIC REGRESSION MODEL  **"         )
print("\n==========================================================================================")
print(model.summary())


############################################################################################
#
#                         TESTING THE LOGISTIC REGRESSION MODEL
#
############################################################################################

# Performing the model test
print("\n==========================================================================================")
print("\n        ** RESULTS OBTAINED FOR TESTING THE LOGISTIC REGRESSION MODEL  **"          )
print("\n==========================================================================================")

# Creating the test dataframe
df_test = pd.DataFrame({'test_results': [1, 1, 0, 0, 1, 1, 0, 0, 1, 0],
                   'study_hours': [2, 2, 4, 4, 4, 3, 2, 2, 4, 5],
                   'study_methods': ['A', 'B', 'B', 'A', 'B', 'B', 'B', 'A', 'B', 'A']})

# Identifying the independent and dependent variables
X_test = df_test[['study_hours', 'study_methods']]
y_test = df_test['test_results']

# Performing the prediction process on the test data
y_predicted = model.predict(X_test)
y_predicted_aprox = list(map(round, y_predicted))

# Showing the result for predictions on the test data,
# as well as the test values ​​themselves for comparison
print("\n Values ​​used in model testing: ", list(y_test.values))
print("\n Predictions made by the model:      ", y_predicted_aprox)


############################################################################################
#
#                CHECKING THE PERFORMANCE (QUALITY) OF THE MODEL
#
############################################################################################
# To check the model's performance, we can create a Confusion Matrix to check
# the number of correct and incorrect predictions it made.
# 
############################################################################################
print("\n==========================================================================================")
print("\n                         ** CONFUSION AND ACCURACY MATRIX  **"                             )
print("\n==========================================================================================")


Confusion_Matrix = pd.crosstab(y_predicted_aprox, y_test, rownames=['predicted'], colnames=['Real'], margins= False)

# Displays the generated Confusion Matrix
print("\n Confusion Matrix: \n", Confusion_Matrix)

# Transforming the confusion matrix so we can visualize and interpret it as a heat map
sn.heatmap(Confusion_Matrix, annot = True)

# Showing the confusion matrix with the matplotlib library
plt.show()

Confusion_Matrix = metrics.confusion_matrix(y_test, y_predicted_aprox)

Recall_Sensitivity = metrics.recall_score(y_test, y_predicted_aprox)
print("\n Sensitivity (Recall): ", Recall_Sensitivity)

Specificity = metrics.recall_score(y_test, y_predicted_aprox, pos_label = 0)
print("\n Specificity: ", Specificity)

Accuracy = metrics.accuracy_score(y_test, y_predicted_aprox)
print("\n Accuracy: ", Accuracy)

Precision = metrics.precision_score(y_test, y_predicted_aprox)
print("\n Precision: ", Precision)


F1_Score = metrics.f1_score(y_test, y_predicted_aprox)
print("\n F1-Score: ", F1_Score)


############################################################################################
#
#                          FORECAST FOR NEW INPUT VALUES
#
############################################################################################

# Performing the forecast for new data
print("\n==========================================================================================")
print("\n                           ** FORECAST FOR NEW DATA  **"                               )
print("\n==========================================================================================")

# Creating the dataframe for forecasting new data
x_new_data = pd.DataFrame({'study_hours':   [ 2,   2,   2,   3 ],
                              'study_methods': ['B', 'A', 'B', 'A']})

# Performing the prediction process on new data
y_predicted_new_data = model.predict(x_new_data)
y_predicted_new_data_aprox = pd.DataFrame(list(map(round, y_predicted_new_data)))


# Generates a new dataset to be presented
# containing the predictor variables along with the data predicted by the model
df_output = [x_new_data['study_hours'], x_new_data['study_methods'], y_predicted_new_data_aprox]

column_names = ['', '', 'Approved?']
# Generates a new dataset (dataframe) by concatenating the dataframes
df_presentation = pd.concat(df_output, axis = 1, keys = column_names)

# Displaying the result for predictions on new data
print("\n Prediction result for new data:")
print("\n", df_presentation)


print("\n The predictions obtained are fractional values ​​(between 0 and 1 or 0% and 100%)") 
print("\n which denote the probability of being approved. These values ​​are therefore") 
print("\n rounded, to obtain discrete values ​​of 1 (passed) or 0 (failed).\n")

############################################################################################
#
#                           INTERPRETATION OF RESULTS
#
############################################################################################

# In the output, ‘Iterations‘ refers to the number of times the model iterates over the data,
# attempting to optimize the model. In this example we had 5 iterations.
# By default, the maximum number of iterations performed is 35, after which the optimization fails.

# The values ​​in the coef column of the output tell us the average change in the log odds
# of passing the exam.

# For example:

# Using study method B is associated with an average increase of 0.0875 in the log odds of passing
# the exam compared to using study method A.
#
# Each additional hour studied is associated with an average increase of 0.4909 in the odds of passing the exam.
# The values ​​in the P>|z| column represent the p-values ​​for each coefficient.

# For example:

# Study methods have a p-value of 0.934. Since this value is not less than 0.05,
# means that there is no statistically significant relationship between study methods and whether
# a student passes the exam or not.
#
# Hours studied has a p-value of 0.045. Since this value is less than 0.05,
# means that there is a statistically significant relationship between hours studied
# and whether a student passes the exam or not.
#
# To assess the quality of the logistic regression model, we can use the
# performance metric in the output below:

# Pseudo R-Squared:

# This value can be considered the surrogate for the R-squared value for a linear regression model.
# It is calculated as the ratio of the maximized log-likelihood function of the null model to the full model.
# This value can range from 0 to 1, with higher values ​​indicating a better fit of the model. 
# # In this example, the pseudo R-squared value is 0.1894, which is a low value. This tells us that
# the predictor variables in the model do not do a very good job of predicting the value of the response variable.

# Log-Likelihood :
#
# The natural logarithm of the Maximum Likelihood Estimation (MLE) function. MLE is the optimization process of finding
# the set of parameters that results in the best fit.

# LL-Null :
#
# The log-likelihood value of the model when no independent variables are included (x1, x2,...),
# that is, only the beta_0 term is taken into account (only the intercept is included) in the logistic equation.
#
#                       P(x) = 1 / (1 - e^(beta_0)) ou ln[ P(x) / ( 1 - P(x))] = beta_0
#
#
