#######################################################################################################################################

# Computation of the Mode ("Moda") for the data on the tables below:

# Recall that the Mode is the value which occurs most often in an array (table, sample or finite population).

# Note that it is possible for an array to contain one or more modes.

#######################################################################################################################################

import numpy as np

def Find_Mode(a):
           
    vals, counts = np.unique(a, return_counts = True)
    
    #find number of occurrences for elements in a data row/column
    occurs = np.argwhere(counts == np.max(counts))
    
    Mode = vals[occurs]
    
    return Mode


def Find_Number_of_Occurrences_of_Modes(a):
    
    vals, counts = np.unique(a, return_counts = True)
    
    #find number of occurrences for elements in a data row/column
    occurs = np.argwhere(counts == np.max(counts))
    
    Mode = vals[occurs]    
    
    #find number of occurrences for elements in a data row/column
    Number_of_Occurrences = np.max(counts).flatten().tolist()
    
    return Number_of_Occurrences
    

Lengths_of_the_Screws_From_Supplier_A = (81  ,    70  ,    83   ,   72  ,    78   ,   81   ,   81  ,    80   ,   80   ,  79)   # Array A

Lengths_of_the_Screws_From_Supplier_B = (80  ,    80  ,    80   ,   79  ,    79   ,   78   ,   81   ,   81   ,   80   ,   81)  # Array B


# Find the Mode elements and how often they occurs for supplier A

Mode_A = Find_Mode(Lengths_of_the_Screws_From_Supplier_A)

Number_of_Occurrences_for_Mode_A = Find_Number_of_Occurrences_of_Modes(Lengths_of_the_Screws_From_Supplier_A)

print("\n Mode for supplier A = ", Mode_A)

print("\n Number of occurrences of the mode for supplier A = ", Number_of_Occurrences_for_Mode_A)


# Find the Mode elements and how often they occurs for supplier B

Mode_B = Find_Mode(Lengths_of_the_Screws_From_Supplier_B)

Number_of_Occurrences_for_Mode_B = Find_Number_of_Occurrences_of_Modes(Lengths_of_the_Screws_From_Supplier_B)

print("\n Mode for supplier B = ", Mode_B)

print("\n Number of occurrences of the mode for supplier B = ", Number_of_Occurrences_for_Mode_B)

