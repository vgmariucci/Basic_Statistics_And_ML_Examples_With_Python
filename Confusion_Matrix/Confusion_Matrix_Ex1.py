############################################################################################################
# Example Statement:

# Suppose you are using a machine learning algorithm that needs to predict whether or not patients
# in a clinic are infected by a virus. After training your algorithm with training data, you
# choose 10 test values ​​and build the following table:

# Prediction | Actual
# ----------------------------
# Has Virus | Has Virus
# No Virus | No Virus
# Has Virus | Has Virus
# Has Virus | Has Virus
# No Virus | Has Virus
# Has Virus | No Virus
# Has Virus | Has Virus
# Has Virus | No Virus
# Has Virus | Has Virus
# Has Virus | No Virus
# Has Virus | Has Virus
# Has Virus | No Virus
# Has Virus | Has Virus
# Has Virus | Has Virus

# An example of how to build a confusion matrix and how to calculate the following parameters:

# - Accuracy;
# - Sensitivity or Recall;
# - Specificity;
# - Accuracy;
# - F1-Score (Harmonic Mean between Precision and Recall)
#
############################################################################################################

# Importing the base libraries
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

# Importing libraries for calculating parameters related to the confusion matrix
from sklearn import metrics

# Table or dataset with test results
test_data = {
'Virus_Predicted_Result': [
"Has Virus",
"No Virus",
"Has Virus",
"Has Virus",
"No Virus", 
"Has Virus",
"Has Virus",
"Has Virus",
"Has Virus",
"Has Virus",
"Has Virus",
"Has Virus",
"Has Virus",
"Has Virus"
]
,
'Virus_Real_Result': 
[
"Has Virus",
"No Virus",
"Has Virus",
"Has Virus",
"Has Virus",
"No Virus",
"Has Virus",
"No Virus",
"Has Virus",
"No Virus",
"Has Virus",
"No Virus",
"Has Virus",
"Has Virus"
]
}

df = pd.DataFrame(data = test_data, columns = ['Virus_Predicted_Result', 'Virus_Real_Result'])
print(df)

# Mapping the problem classifiers
# Has Virus = 1
# No Virus = 0
df['Virus_Predicted_Result'] = df['Virus_Predicted_Result'].map({'Has Virus': 1, 'No Virus': 0})
df['Virus_Real_Result'] = df['Virus_Real_Result'].map({'Has Virus': 1, 'No Virus': 0})

# Creating the confusion matrix for the test data
confusion_matrix = pd.crosstab(df['Virus_Predicted_Result'], df['Virus_Real_Result'], rownames = ['Virus_Predicted_Result'], colnames=['Virus_Real_Result'], margins = False)
print(confusion_matrix)


# Transforming the confusion matrix so we can visualize and interpret it as a heat map
sn.heatmap(confusion_matrix, annot = True)

#Showing the confusion matrix with the matplotlib library
plt.show()

# ############################################################################################################
# #
# # Calculating and presenting the parameters: Recall, Specificity, Accuracy, Precision and F1-Score
# #
# ###########################################################################################################

confusion_matrix = metrics.confusion_matrix(df['Virus_Predicted_Result'], df['Virus_Real_Result'])


#######################################################################################################################################
#
# For the correct calculation of the parameters it was necessary to invert the position of each data list in the dataset, so that:
#  
#   df['Virus_Predicted_Result'], df['Virus_Real_Result'] --> df['Virus_Real_Result'], df['Virus_Predicted_Result']
#
########################################################################################################################################
Recall_Sensitivity = metrics.recall_score(df['Virus_Real_Result'], df['Virus_Predicted_Result'])
print("\n Sensibilidade (Recall): ", Recall_Sensitivity)

Specificity = metrics.recall_score(df['Virus_Real_Result'], df['Virus_Predicted_Result'], pos_label = 0)
print("\n Specificity: ", Specificity)

Accuracy = metrics.accuracy_score(df['Virus_Real_Result'], df['Virus_Predicted_Result'])
print("\n Accuracy: ", Accuracy)

Precision = metrics.precision_score(df['Virus_Real_Result'], df['Virus_Predicted_Result'])
print("\n Precision: ", Precision)


F1_Score = metrics.f1_score(df['Virus_Real_Result'], df['Virus_Predicted_Result'])
print("\n F1-Score: ", F1_Score)





