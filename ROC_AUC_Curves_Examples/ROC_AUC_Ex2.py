##########################################################################################################################
# 
# It is often interesting to fit several classification models to a data set, creating ROC curves for
# each model and checking which one performs best in predicting the data.
#
##########################################################################################################################

# Importing the libraries
from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import matplotlib.pyplot as plt

# Next, we will use sklearn's make_classification() function to create a fake dataset with 1,000 rows,
# four predictor variables and one binary response variable:

# Creating the dataset with 1,000 rows
X, y = datasets.make_classification( n_samples = 1000,
                                     n_features = 4,
                                     n_informative = 3,
                                     n_redundant = 1,
                                     random_state = 0)

# Separate data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 0)

# Next, we will fit a logistic regression model and then a gradient boosted model
# to the data and plot the ROC curve for each model on the same graph:

# Configure the figure that will contain the ROC curve graphs

# Perform the training and testing processes using Logistic Regression and then build the ROC curve
Reg_Log_Model = LogisticRegression()
Reg_Log_Model.fit(X_train, y_train)
y_predicted = Reg_Log_Model.predict_proba(X_test)[:, 1]
false_positive_rate, true_positive_rate, _ = metrics.roc_curve(y_test, y_predicted)
auc = round(metrics.roc_auc_score(y_test, y_predicted), 4)
plt.plot(false_positive_rate, true_positive_rate, label = "Logistig Regresssion -> AUC = " + str(auc))

# Performs the training and testing processes using Gradient Boosted and then builds the ROC curve
Grad_Boosted_Model = GradientBoostingClassifier()
Grad_Boosted_Model.fit(X_train, y_train)
y_predicted = Grad_Boosted_Model.predict_proba(X_test)[:, 1]
false_positive_rate, true_positive_rate, _ = metrics.roc_curve(y_test, y_predicted)
auc = round(metrics.roc_auc_score(y_test, y_predicted), 4)
plt.plot(false_positive_rate, true_positive_rate, label = "Gradiente Boosting -> AUC = " + str(auc))

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# The blue line shows the ROC curve for the logistic regression model and the orange line shows the ROC curve for the gradient boosted 
# model.

# The more the ROC curve has a kink in the upper left corner of the graph, the better the model is able to classify the data into its 
# respective categories.

# To quantify this, we can calculate the AUC – area under the curve – which tells us how much of the graph is located under the curve.

# The closer the AUC is to 1, the better the model.

# In our graph, we can see the following AUC metrics for each model:

# Logistic regression model AUC: 0.7902
# Gradient boosted model AUC: 0.9712

# Clearly, the gradient boosted model does a better job of classifying the data into categories
# compared to the logistic regression model.