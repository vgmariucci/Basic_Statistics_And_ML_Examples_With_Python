################################################################################################################
# In statistics, the range is important for the following reasons:

# Reason 1: It tells the spread of the entire dataset.

# Reason 2: It tells what extreme values are possible in a given dataset.

# The Drawback of Using the Range:

# The range suffers from one drawback: It is influenced by outliers.

#################################################################################################################

import pandas as pd
import numpy as np

Lengths_of_the_Screws_From_Supplier_A = (81  ,    70  ,    83   ,   72  ,    78   ,   81   ,   81  ,    80   ,   80   ,  79)   # Array A

Lengths_of_the_Screws_From_Supplier_B = (80  ,    80  ,    80   ,   79  ,    79   ,   78   ,   81   ,   81   ,   80   ,   81)  # Array B


def Range_Calculation(a):
    
    Max = max(a)
    
    Min = min(a)
    
    Range = Max - Min

    return Range


Range_A = Range_Calculation(Lengths_of_the_Screws_From_Supplier_A)

print("Range for supplier A: ", Range_A)

Range_B = Range_Calculation(Lengths_of_the_Screws_From_Supplier_B)

print("Range for supplier B: ", Range_B)
