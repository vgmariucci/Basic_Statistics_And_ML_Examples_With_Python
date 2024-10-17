##########################################################################################################################
#
# Calculating the standard deviation for the ages of The Simpsons characters
#
# Calculating the population and sample standard deviation for the ages of The Simpsons characters:

# The basic rules for calculating standard deviations are:

# * We calculate the population standard deviation when the data set is the entire population.

# * We consider the sample standard deviation if our data sets represent a sample taken from a large population (as is the case for the ages of The Simpsons characters). #
# NOTE:
# The sample standard deviation will always be larger than the population standard deviation for
# the same data set because there is more uncertainty in calculating the sample standard deviation,
# so our estimate of the standard deviation will be larger.
#
##########################################################################################################################

import pandas as pd

dataset = {'Group 1': [1, 8, 10, 38, 39],
         'Group 2': [8, 10, 39, 45, 49]}
dataframe = pd.DataFrame(data = dataset)

print("\n Average: ", dataframe['Group 1'].mean())
print("\n Median: ", dataframe['Group 1'].median())
print("\n Mode: ", dataframe['Group 1'].mode())
print("\n Standard Deviation for Group 1: ", dataframe['Group 1'].std()) 

print("\n Average: ", dataframe['Group 2'].mean())
print("\n Median: ", dataframe['Group 2'].median())
print("\n Mode: ", dataframe['Group 2'].mode())
print("\n Standard Deviation for Group 2: ", dataframe['Group 2'].std())

