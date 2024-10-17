#################################################################################################################
#
#  Determination of the Median
#
# Recall that the median represents the value that lies directly in the middle of a dataset, 
# when all of the values are arranged from smallest to largest.

# The median represents the middle value of a dataset, when all of the values 
# are arranged from smallest to largest.

# For example, the median in the following dataset is 19:

# Dataset: 3, 4, 11, 15, 19, 22, 23, 23, 26

# The median also represents the 50th percentile of a dataset. That is, exactly half of the values 
# in the dataset are larger than the median and half of the values are lower.

# The median is an important metric to calculate because it gives us an idea of where the “center” 
# of a dataset is located. It also gives us an idea of the “typical” value in a given dataset.

# When to Use the Median?

# When analyzing datasets, we’re often interested in understanding where the center value is located.

# In statistics, there are two common metrics that we use to measure the center of a dataset:

# Mean: The average value in a dataset
# Median: The middle value in a dataset

# It turns out that the median is a more useful metric in the following circumstances:

# When the distribution is skewed.
# When the distribution contains outliers.

# Summary

# The median represents the middle value in a dataset.
# The median is important because it gives us an idea of where the center value is located in a dataset.
# The median tends to be more useful to calculate than the mean when a distribution is skewed and/or has outliers.
################################################################################################################

import numpy as np

Lengths_of_the_Screws_From_Supplier_A = (81  ,    70  ,    83   ,   72  ,    78   ,   81   ,   81  ,    80   ,   80   ,  79)   # Array A

Lengths_of_the_Screws_From_Supplier_B = (80  ,    80  ,    80   ,   79  ,    79   ,   78   ,   81   ,   81   ,   80   ,   81)  # Array B

def median_calculation(a):
    Median = np.median(a)
    
    return Median


Median_A = median_calculation(Lengths_of_the_Screws_From_Supplier_A)

print("Mendian for supplier A: ",Median_A)

Median_B = median_calculation(Lengths_of_the_Screws_From_Supplier_B)

print("Mendian for supplier B: ",Median_B)