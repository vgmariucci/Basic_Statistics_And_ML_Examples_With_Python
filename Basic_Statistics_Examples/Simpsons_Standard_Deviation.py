############################################################################################
#
# Calculating the standard deviation for the ages of The Simpsons characters
#
# Calculating the population and sample standard deviation for the ages of The Simpsons characters:

# The basic rules for calculating standard deviations are:

# * We calculate the population standard deviation when the data set is the entire population.

# * We consider the sample standard deviation if our data sets represent a sample taken from a large population (as is the case for the ages of The Simpsons characters).
#
# NOTE:
# The sample standard deviation will always be larger than the population standard deviation for
# the same data set because there is more uncertainty in calculating the sample standard deviation,
# so our estimate of the standard deviation will be larger.
###########################################################################################

import statistics as stat
import numpy as np


group_1 = (1, 8, 10, 38, 39)

group_2 = (8, 10, 39, 45, 49)



# Calculation of population and sample standard deviations using the Numpy library
def Calculate_Population_Standard_Deviation_With_Numpy(a):
    
    Population_STD_Numpy = np.std(a)
    
    return Population_STD_Numpy

def Calculate_Sample_Standard_Deviation_With_Numpy(a):
    
    Sample_STD_Numpy = np.std(a, ddof = 1)
    
    return Sample_STD_Numpy


# Calculation of population and sample standard deviations using the Statistics library
def Calculate_Population_Standard_Deviation_Stat(a):
    
    Population_STD_Stat = stat.pstdev(a)
    
    return Population_STD_Stat


def Calculate_Sample_Standard_Deviation_Stat(a):
    
    Sample_STD_Stat = stat.stdev(a)
    
    return Sample_STD_Stat


print("\n Population Standard Deviation for group 1 (using numpy): ", Calculate_Population_Standard_Deviation_With_Numpy(group_1))

print("\n Population Standard Deviation for group 2 (using numpy): ", Calculate_Population_Standard_Deviation_With_Numpy(group_2))



print("\n Sample Standard Deviation for group 1 (using numpy): ", Calculate_Sample_Standard_Deviation_With_Numpy(group_1))

print("\n Sample Standard Deviation for group 2 (using numpy): ", Calculate_Sample_Standard_Deviation_With_Numpy(group_2))



print("\n Population Standard Deviation for group 1 (using statistics): ", Calculate_Population_Standard_Deviation_Stat(group_1))

print("\n Population Standard Deviation for group 2 (using statistics): ", Calculate_Population_Standard_Deviation_Stat(group_2))



print("\n Sample Standard Deviation for group 1 (using statistics): ", Calculate_Sample_Standard_Deviation_Stat(group_1))

print("\n Sample Standard Deviation for group 2 (using statistics): ", Calculate_Sample_Standard_Deviation_Stat(group_2))

