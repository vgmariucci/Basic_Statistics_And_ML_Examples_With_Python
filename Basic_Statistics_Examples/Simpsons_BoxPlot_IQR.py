############################################################################################
# 
# Determining the Interquartile Range (IQR) and Presenting the Boxplot for the Ages of The Simpsons Characters
#
# A box plot is a type of graph that displays the five-number summary of a data set, which includes:

# The minimum value
# The first quartile (the 25th percentile)
# The median value
# The third quartile (the 75th percentile)
# The maximum value

# We use the following process to draw a box plot:

# 1- Draw a box from the first quartile (Q1) to the third quartile (Q3)
# 2- Draw a line inside the box at the median
# 3- Draw “whiskers” between the Q1 and Q3 quartiles for the minimum and maximum values ​​respectively

# When the median is closer to the bottom of the box
# and the whisker is shorter at the lower or left end of the box,
# the distribution is skewed to the right (or asymmetric “positively”).

# When the median is closer to the top of the box
# and the whiskers are shorter at the upper or right end of the box,
# the distribution is skewed to the left (or “negatively” skewed).

# When the median is in the middle of the box
# and the whiskers are approximately equal on each side,
# the distribution is symmetric (or “unskewed”).

###########################################################################################

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Set the figure size
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

dados = {'Group 1': [1, 8, 10, 38, 39],
         'Group 2': [8, 10, 39, 45, 49]}

dataframe = pd.DataFrame(data = dados)


# Plot the dataframe
boxplot = dataframe[['Group 1', 'Group 2']].plot(kind='box', title='boxplot')
plt.show()


################################################# #############################
#
# DETERMINATION OF THE FOLLOWING VALUES:
#
# 1ST QUARTILE;
# 2ND QUARTILE (median);
# 3RD QUARTILE;
# 4TH QUARTILE;
# IQR;
# MINIMUM VALUE;
# MAXIMUM VALUE;
#
################################################# #############################

q3_Group_1, q1_Group_1 = np.percentile(dataframe['Group 1'], [75, 25])

IQR_Group_1 = q3_Group_1 - q1_Group_1


print("\n Group 1")
print("\n Idade máxima: ", max(dataframe['Group 1']))
print("\n Idadae Mínima: ", min(dataframe['Group 1']))
print("\n q1: ", q1_Group_1)
print("\n q3: ", q3_Group_1)
print("\n IQR: ", IQR_Group_1)


q3_Group_2, q1_Group_2 = np.percentile(dataframe['Group 2'], [75, 25])

IQR_Group_2 = q3_Group_2 - q1_Group_2

print("\n Group 2")
print("\n Idade máxima: ", max(dataframe['Group 2']))
print("\n Idadae Mínima: ", min(dataframe['Group 2']))
print("\n q1: ", q1_Group_2)
print("\n q3: ", q3_Group_2)
print("\n IQR: ", IQR_Group_2)





