#######################################################################################################################################

# Computation of the Arithmetic Mean or Average for the data on tables above:

#######################################################################################################################################
import pandas as pd
import numpy as np

Lengths_of_the_Screws_From_Supplier_A = (81  ,    70  ,    83   ,   72  ,    78   ,   81   ,   81  ,    80   ,   80   ,  79)   # Array A

Lengths_of_the_Screws_From_Supplier_B = (80  ,    80  ,    80   ,   79  ,    79   ,   78   ,   81   ,   81   ,   80   ,   81)  # Array B

Average_Length_for_Screws_From_Supplier_A = np.mean(Lengths_of_the_Screws_From_Supplier_A)

Average_Length_for_Screws_From_Supplier_B = np.mean(Lengths_of_the_Screws_From_Supplier_B)

print ("\n <Len_A> = ", Average_Length_for_Screws_From_Supplier_A)

print ("\n <Len_B> = ", Average_Length_for_Screws_From_Supplier_B)







