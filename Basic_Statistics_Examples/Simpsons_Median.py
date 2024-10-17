############################################################################################
#
#  Determining the Median Age of The Simpsons Characters
# 
###########################################################################################

import pandas as pd
import numpy as np

group_1 = (1, 8, 10, 38, 39)

group_2 = (8, 10, 39, 45, 49)

def compute_median(a):
    
    median = np.median(a)
    
    return median


median_group_1 = compute_median(group_1)

median_group_2 = compute_median(group_2)

print("Median for group ages 1: ", median_group_1)

print("Median for group ages 2: ", median_group_2)