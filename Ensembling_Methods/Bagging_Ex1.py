#############################################################################################################
#
# BOOTSTRAP AGGREGATION (BAGGING)

# Bootstrap Aggregating (Bagging) is an ensemble method commonly used to optimize machine learning models and overcome overfitting effects for classification or regression problems.

# Bagging aims to improve the accuracy and performance of machine learning algorithms.

# It does this by taking random subsets of an original dataset, with replacement,

# and fitting a classifier (for classification) or regressor (for regression) to each subset.

# The predictions for each subset are aggregated via majority voting for classification or averaged for regression, increasing the accuracy of the model.

# We will seek to identify different classes of wines found in the Sklearn wine dataset.
#
###############################################################################################################

# Importando as bibliotecas
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Next, we need to load the data and store it in X (input features) and y (output or response).
# The as_frame parameter is chosen as True so that we don't lose the feature names when loading the data.
# (sklearn version prior to 0.23 should ignore the as_frame argument as it is not supported)

data = datasets.load_wine(as_frame = True)

# print(data)

X = data.data
# print(X)

y = data.target
# print(y)

# To properly evaluate our model on unseen data, we need to split X and y into training and testing sets.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 22)

# With our data prepared, we can now instantiate a base classifier and fit it to the training data.
dtree = DecisionTreeClassifier(random_state = 22)
dtree.fit(X_train, y_train)


# Now we can predict the wine class from the unseen test set and evaluate the model performance.
y_predicted = dtree.predict(X_test)

print("\n Accuracy obtained with training data: ", accuracy_score(y_true = y_train, y_pred = dtree.predict(X_train)))
print("\n Accuracy obtained with test data: ", accuracy_score(y_true = y_test, y_pred = y_predicted))

# The base classifier performs reasonably well on the dataset, achieving 82% accuracy on the test dataset
# with the current parameters (different results may occur if you do not have the random_state parameter set).

# Now that we have a baseline accuracy for the test dataset, we can see how the Bagging Classifier
# performs on a single decision tree classifier.

############################################################################################
#
#                           CREATING A BAGGING SORTER
#
############################################################################################

# For bagging, we need to define the parameter n_estimators, this is the number of base classifiers that our model will aggregate.
# For this dataset the number of estimators is relatively low, usually much larger ranges are used
# Hyperparameter tuning is usually done with a grid search, but for now we will use a selected set of
# values ​​for the number of estimators.

# We start by importing the necessary model:
from sklearn.ensemble import BaggingClassifier

# Now let's create a range of values ​​that represent the number of estimators we want to use
# in each ensemble.

estimator_interval = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
 
# To see how Classified Bagging works with different estimator values, we need a way to iterate
# over the range of values ​​and store the results for each ensemble. To do this, we will create a for loop,
# storing the models and scores in separate lists for later visualization.

# Note: The default parameter for the base classifier in BaggingClassifier is DicisionTreeClassifier,
# so we don't need to define it when instantiating the Bagging model.

models =[]
points = []

for n_estimators in estimator_interval:
    
    # Creation of the Bagging classifier
    clf = BaggingClassifier(n_estimators = n_estimators, random_state = 22)
    
    # Performing the adjustment
    clf.fit(X_train, y_train)
    
    # Records the model number tested and its respective score
    models.append(clf)
    points.append(accuracy_score(y_true = y_test, y_pred = clf.predict(X_test)))
    
# With the models and scores stored, we can now visualize the improvement in model performance.
    
# Construction of the graph of scores as a function of the number of estimators
plt.figure(figsize=(9, 6))
plt.plot(estimator_interval, points)
plt.xlabel("Estimator number", fontsize = 18)
plt.ylabel("Scoring", fontsize = 18)
plt.tick_params(labelsize = 16)
plt.show()

# Interpretation of Results

# By iterating through different values ​​for the number of estimators, we can see an increase in model performance from
# 82% to 95%. After 15 estimators, the accuracy starts to drop again, if you set a different random_state,
# you will observe different values. This is why it is a good practice to use cross-validation to ensure stable results.

# In this case, we see a 13% increase in accuracy for identifying the type of wine.

#############################################################################################################
#
#                   GENERATING DECISION TREES FROM THE BAGGING CLASSIFIER
#
#############################################################################################################

# You can see the individual decision trees that went into the aggregated classifier.
# This helps us get a more intuitive understanding of how the bagging model arrives at its results.
# NOTE: This is only functional with smaller datasets, where the trees are relatively small,
# making it easier to visualize.


clf = BaggingClassifier(n_estimators = 13, oob_score = True, random_state = 22)

clf.fit(X_train, y_train)

plt.figure(figsize = (10, 10))

plot_tree(clf.estimators_[0], feature_names = X.columns)

plt.show()

# We can see the decision trees that were generated in the model.
# Changing the classifier index.