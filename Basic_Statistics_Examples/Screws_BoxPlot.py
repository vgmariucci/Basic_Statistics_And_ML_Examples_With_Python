################################################################################################################
#
# Determination of the Interquartile Range (IQR) and Boxplot presentation
# 
# A box plot is a type of plot that displays the five number summary of a dataset, which includes:

# The minimum value
# The first quartile (the 25th percentile)
# The median value
# The third quartile (the 75th percentile)
# The maximum value
# We use the following process to draw a box plot:

# Draw a box from the first quartile (Q1) to the third quartile (Q3)
# Then draw a line inside the box at the median
# Then draw “whiskers” from the quartiles to the minimum and maximum values

# When the median is closer to the bottom of the box 
# and the whisker is shorter on the lower end of the box, 
# the distribution is right-skewed (or “positively” skewed).

# When the median is closer to the top of the box 
# and the whisker is shorter on the upper end of the box, 
# the distribution is left-skewed (or “negatively” skewed).

# When the median is in the middle of the box 
# and the whiskers are roughly equal on each side, 
# the distribution is symmetrical (or “no” skew).
#################################################################################################################

import pandas as pd
import numpy as np


Lengths_of_the_Screws_From_Supplier_A = (81  ,    70  ,    83   ,   72  ,    78   ,   81   ,   81  ,    80   ,   80   ,  79)   # Array A

Lengths_of_the_Screws_From_Supplier_B = (80  ,    80  ,    80   ,   79  ,    79   ,   78   ,   81   ,   81   ,   80   ,   81)  # Array B

################################################################################
#   OBS: 
# THE CODES BELOW DON'T WORKED WELL WITH VSCODE (THE BOXPLOTS ARE NOT SHOWN).
# 
# BUT WITH ANACONDA SPYDER IT WORKED NICELY
###############################################################################

# Create a dataframe before the construction of boxplots
df = pd.DataFrame({'Screws From supplier A': Lengths_of_the_Screws_From_Supplier_A,
                    'Screws From supplier B': Lengths_of_the_Screws_From_Supplier_B})

# View dataframe
print(df)

df.boxplot(column=['Screws From supplier A','Screws From supplier B'], grid = False, color = 'black')


###############################################################################
#
# DETERMINATION OF THE FOLLOWING VALUES:
# 
# 1st QUARTILE;
# 2nd QUARTILE(median);
# 3rd QUARTILE; 
# 4th QUARTILE;
# IQR;
# MINIMUM VALUE;
# MAXIMUM VALUE;
#
###############################################################################

q3A_Numpy, q1A_Numpy = np.percentile(Lengths_of_the_Screws_From_Supplier_A, [75, 25])

IQR_A_Numpy = q3A_Numpy - q1A_Numpy

print("\n ========================================================================")
print("\n The values below were obtained using numpy")
print("\n ========================================================================")

print("\n Supplier A")
print("\n Max_Value_A: ", max(Lengths_of_the_Screws_From_Supplier_A))
print("\n Min_Value_A: ", min(Lengths_of_the_Screws_From_Supplier_A))
print("\n q1A_Numpy: ", q1A_Numpy)
print("\n q3A_Numpy: ", q3A_Numpy)
print("\n IQR_A_Numpy: ", IQR_A_Numpy)


q3B_Numpy, q1B_Numpy = np.percentile(Lengths_of_the_Screws_From_Supplier_B, [75, 25])

IQR_B_Numpy = q3B_Numpy - q1B_Numpy
print("\n Supplier B")
print("\n Max_Value_B: ", max(Lengths_of_the_Screws_From_Supplier_B))
print("\n Min_Value_B: ", min(Lengths_of_the_Screws_From_Supplier_B))
print("\n q1B_Numpy: ", q1B_Numpy)
print("\n q3B_Numpy: ", q3B_Numpy)
print("\n IQR_B_Numpy: ", IQR_B_Numpy)

###############################################################################
# OBS:These parameters calculated by pandas library gave different results when
#     compared to the method use by:
#   
#   view-source:https://www.statology.org/boxplot-generator/

# Next, we use the same method used in the statology.org site to determine
# the above parameters 
##############################################################################

# Determine the first quartile (q1) of the dataset array passed in variable a 
def first_quartile(a):
    
    #This array will be filled with the values smaller than median_a value
    first_half_array = [] 
    
    #Determine the median value for the original array
    median_a = np.median(a)
    
    #Fill the first_half_array with the values of the original array which are smaller than median_a
    for  i in a:
        if i < median_a: 
            first_half_array.append(i)
                
    q1 = np.median(first_half_array)
    
    #Return the first quartile q1
    return q1 




# Determine the third quartile (q3) of the dataset array passed in variable a
def third_quartile(a):
    
    #This array will be filled with the values greater than median_a value
    second_half_array = [] 
    
    #Determine the median value for the original array
    median_a = np.median(a)
    
    #Fill the second_half_array with the values of the original array which are greater than median_a
    for  i in a:
        if i > median_a: 
            second_half_array.append(i)
                
    q3 = np.median(second_half_array)
    
    # Return the third quartile q3
    return q3 


#Determine the Interquartile Range (IQR)
def IQR_calc(a):
                   
    q1 = first_quartile(a)
        
    q3 = third_quartile(a)
        
    IQR = q3 - q1
        
    return IQR   

print("\n ========================================================================")
print("\n The values below were obtained with the method used in statology.org")
print("\n ========================================================================")

print("\n Supplier A dataset")
print("\n Max_Value_A: ", max(Lengths_of_the_Screws_From_Supplier_A))
print("\n Min_Value_A: ", min(Lengths_of_the_Screws_From_Supplier_A))
print("\n q1_A: ", first_quartile(Lengths_of_the_Screws_From_Supplier_A))
print("\n q3_A: ", third_quartile(Lengths_of_the_Screws_From_Supplier_A))
print("\n IQR_A: ", IQR_calc(Lengths_of_the_Screws_From_Supplier_A))

print("\n Supplier B dataset")
print("\n Max_Value_B: ", max(Lengths_of_the_Screws_From_Supplier_B))
print("\n Min_Value_B: ", min(Lengths_of_the_Screws_From_Supplier_B))
print("\n q1_B: ", first_quartile(Lengths_of_the_Screws_From_Supplier_B))
print("\n q3_B: ", third_quartile(Lengths_of_the_Screws_From_Supplier_B))
print("\n IQR_B: ", IQR_calc(Lengths_of_the_Screws_From_Supplier_B))