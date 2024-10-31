###########################################################################################################
#
# Most machine learning models contain parameters that can be tuned to vary
# how the model learns. For example, the sklearn logistic regression model has a parameter C
# that controls regularization, which affects the complexity of the model.
#
# How do we choose the best value for C?
# The best value depends on the data used to train the model.
#
#############################################################################################################

# How does it work?
#
# One method is to try different values ​​and then choose the value that gives the best score.
# This technique is known as grid search. If we were to select values
# for two or more parameters, we would evaluate all combinations of the sets of values, thus forming
# a grid of values.

# Before we get into the example, it's good to know what the parameter we're changing does.
#
# Higher values ​​of C tell the model that the training data resembles real-world information, placing a greater weight on the training data. While lower values ​​of C
# do the opposite.

# First, let's see what kind of results we can generate without a grid search using just
# the basic parameters. To get started, we must first load the dataset we'll be working with.

# Importing the libraries
from sklearn import datasets
from sklearn.linear_model import LogisticRegression


########################################################################
#
#    EXPLORING THE EXISTENCE OF SOME RELATIONSHIP BETWEEN THE DATA
#
########################################################################

# Next, to create the model we need a set of independent variables X
# and a dependent variable y
iris = datasets.load_iris()

X = iris['data']
y = iris['target']


print(X)
print(y)


# We will use the logistic regression method to classify the data

########################################################################
#
#                CONSTRUCTION OF THE LOGISTIC REGRESSION MODEL
#
########################################################################

# In this first step, we will choose the maximum number of iterations to ensure that the regression process
# can obtain an acceptable result.
# Remember that the default value for the C parameter in logistic regression is 1, as we will compare it with other
# values ​​later on.

# In this example, the goal is to analyze the iris data values ​​and try to train the model with different
# values ​​of the C parameter for logistic regression.

logistic_regression_model = LogisticRegression(max_iter = 10000)

# After creating the model, we can try to fit it to the real data
print(logistic_regression_model.fit(X,y))

print(logistic_regression_model.score(X,y))

# With the default setting of C = 1, we achieved a score of 0.973.
# Let's see if we can do better by implementing a grid search with values ​​other than 0.973.

# Implementing the grid search

# We'll follow the same steps as before, except this time we'll define a range of values ​​for C.
# Knowing what values ​​to set for the searched parameters will require a combination of domain knowledge and practice.
# Since the default value for C is 1, we'll define a range of values ​​around it.
C = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

# Next, we will create a for loop to swap the values ​​of C and perform the adjustment using logistic regression for each selected value of C. 
# But first, let's create an empty array to store the score values ​​for each value of C.

scores = []

for i in C:
    logistic_regression_model.set_params(C = i)
    logistic_regression_model.fit(X,y)
    scores.append(logistic_regression_model.score(X,y))

# Once we have recorded the score values ​​(colors) for each C value, we can evaluate which one is the best:
print(scores)

# Explaining the Results

# We can see that lower values ​​of C performed worse than the base parameter of 1.
# However, as we increased the value of C to 1.75, the model experienced increased accuracy.
# Another important observation is that increasing C beyond 1.75 does not result in an increase in the accuracy of the model,
# since when C = 2 the score result remains the same as the value obtained for C = 1.75.


