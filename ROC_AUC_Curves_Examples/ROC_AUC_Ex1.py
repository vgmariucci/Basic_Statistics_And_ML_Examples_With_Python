############################################################################################################################
# Receiver Operating Characteristic (ROC) Curve and Area Under the Curve (AUC)
#
# In classification, there are many different evaluation metrics. The most popular is accuracy, which measures how often the 
# model makes correct predictions. This is a great metric because it is easy to understand and obtain, since the most correct 
# guess is usually desired. There are some cases where you might consider using another evaluation metric.
#
# Another common metric is AUC, the area under the ROC (Receiver Operating Characteristic) curve.
# ROC plots the true positive rate (TP) versus the false positive rate (FP) at different classification thresholds.
# Thresholds are different probability cutoffs that separate the two classes in binary classification.
#
###############################################################################################################################

# Suppose we have an imbalanced dataset where most of our data has only one value.
# We can get high accuracy for the model by predicting the majority class.

# Importing the libraries
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Creating an unbalanced database
n = 1000
proportion = .95
n_A = int((1-proportion)*n)
n_B = int(proportion * n)

y = np.array([0] * n_A + [1] * n_B)

# Below are the probabilities obtained from a hypothetical model that always predicts the majority class
# The probability of predicting class 1 will be 100%
y_prob_1 = np.array([1]*n)
y_pred_1 = y_prob_1 > .5

print(f'Accuracy: {accuracy_score(y, y_pred_1)}')
cf_mat = confusion_matrix(y, y_pred_1)
print('Confusion Matrix')
print(cf_mat)
print(f'Accuracy for class A: {cf_mat[0][0]/n_A}')
print(f'Accuracy for class B: {cf_mat[1][1]/n_B}')

# Although we achieved very high accuracy, the model did not provide any information about the data, so it is not useful.
#
# We accurately predicted class A 100% of the time, while inaccurately predicting class B 0% of the time.
# At the expense of accuracy, it may be better to have a model that can somewhat separate the two classes.

y_prob_2 = np.array(
    np.random.uniform(0, .7, n_A).tolist() + 
    np.random.uniform(.3, 1, n_B).tolist()
)

y_predict_2 = y_prob_2 > .5

print(f'Accuracy: {accuracy_score(y, y_predict_2)}')
cf_mat = confusion_matrix(y, y_predict_2)
print('Confusion Matrix')
print(cf_mat)
print(f'Accuracy da classe A: {cf_mat[0][0]/n_A}')
print(f'Accuracy da classe B: {cf_mat[1][1]/n_B}')

# For the second set of predictions, we don't have as high an accuracy score as the first,
# but the accuracy for each class is more balanced. Using accuracy as an evaluation metric,
# we would rank the first model higher than the second, even though it doesn't tell us anything about the data.

# In cases like this, it would be preferable to use another evaluation metric like AUC.

# Compute the ROC curve based on the probabilities for the model predictions
def compute_ROC_values():
    
    
    # Creating a figure with 2 subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))  # 1 row, 2 columns

    # Construct the ROC plot for model 1
    false_positive_rate, true_positive_rate, _ = roc_curve(y, y_prob_1)
    axes[0].plot(false_positive_rate, true_positive_rate)
    axes[0].set_title('ROC Curve for Model 1')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate') 
    
    # Construct the ROC plot for model 2
    false_positive_rate, true_positive_rate, _ = roc_curve(y, y_prob_2)
    axes[1].plot(false_positive_rate, true_positive_rate)
    axes[1].set_title('ROC Curve for Model 2')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate') 
    
    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the figure
    plt.show()

# Calculate the area under the AUC curve for model 1
print(f'AUC value for model 1: {roc_auc_score(y, y_prob_1)}')


# Calculate the area under the AUC curve for model 2
print(f'AUC value for model 2: {roc_auc_score(y, y_prob_2)}')

# Call the function do calculate and plot ROC curves
compute_ROC_values()


# An AUC score of around 0.5 would mean that the model is unable to distinguish between the two classes
# and the curve would look like a line with a slope of 1.
#
# An AUC score closer to 1 means that the model has the ability to separate the two classes and the curve
# will move closer to the top left corner of the graph.




