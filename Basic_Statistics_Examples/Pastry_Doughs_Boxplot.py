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
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Set the figure size
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True


#Creating the dataframe
data = {'Mixture 1':[22.02   ,   23.83  ,   26.67  ,    25.38  ,    25.49   ,   23.50  ,    25.90   ,  24.89],
         'Mixture 2':[21.49   ,   22.67  ,    24.62 ,     24.18 ,     22.78  ,    22.56 ,     24.46  ,  23.79],
         'Mixture 3':[20.33   ,   21.67  ,    24.67 ,     22.45 ,     22.29  ,    21.95 ,     20.49  ,   21.81]
         }




df = pd.DataFrame(data = data)

# View dataframe
print(df)

# boxplot for dataframe = df
boxplot = df[['Mixture 1', 'Mixture 2', 'Mixture 3']].plot(kind='box', title='boxplot')
plt.show()

# Calculating the first and third quartiles with numpy
q3_Numpy_Mixture1, q1_Numpy_Mixture1 = np.percentile(data["Mixture 1"], [75, 25])

# Calculating the interquartile range (IQR) with numpy
IQR_Numpy_Mixture1 = q3_Numpy_Mixture1 - q1_Numpy_Mixture1

print("\n ========================================================================")
print("\n The values ​​below were obtained using numpy")
print("\n ========================================================================")

print("\n Mixture 1")
print("\n Max Value for Mixture 1: ", max(data["Mixture 1"]))
print("\n Min VAlue for Mixture 1: ", min(data["Mixture 1"]))
print("\n 1st quartile (q1) for Mixture 1: ", q1_Numpy_Mixture1)
print("\n 3rd quartile (q3) for Mixture 1: ", q3_Numpy_Mixture1)
print("\n IQR_Numpy for Mixture 1: ", IQR_Numpy_Mixture1)


# Calculating the first and third quartiles with numpy
q3_Numpy_Mixture2, q1_Numpy_Mixture2 = np.percentile(data["Mixture 2"], [75, 25])

# Calculating the interquartile range (IQR) with numpy
IQR_Numpy_Mixture2 = q3_Numpy_Mixture2 - q1_Numpy_Mixture2


print("\n Mixture 2")
print("\n Max Value for Mixture 2: ", max(data["Mixture 2"]))
print("\n Min VAlue for Mixture 2: ", min(data["Mixture 2"]))
print("\n 1st quartile (q1) for Mixture 2: ", q1_Numpy_Mixture2)
print("\n 3rd quartile (q3) for Mixture 2: ", q3_Numpy_Mixture2)
print("\n IQR_Numpy for Mixture 2: ", IQR_Numpy_Mixture2)

# Calculating the first and third quartiles with numpy
q3_Numpy_Mixture3, q1_Numpy_Mixture3 = np.percentile(data["Mixture 3"], [75, 25])

# Calculating the interquartile range (IQR) with numpy
IQR_Numpy_Mixture3 = q3_Numpy_Mixture3 - q1_Numpy_Mixture3


print("\n Mixture 3")
print("\n Max Value for Mixture 3: ", max(data["Mixture 3"]))
print("\n Min VAlue for Mixture 3: ", min(data["Mixture 3"]))
print("\n 1st quartile (q1) for Mixture 3: ", q1_Numpy_Mixture3)
print("\n 3rd quartile (q3) for Mixture 3: ", q3_Numpy_Mixture3)
print("\n IQR_Numpy for Mixture 3: ", IQR_Numpy_Mixture3)

###############################################################################
# NOTE: The parameters calculated with numpy generally provided different results
# compared to the method used by the website below:
#
# view-source:https://www.statology.org/boxplot-generator/

# Below, we use the same method used on the statology.org website to obtain the
# parameters obtained with numpy
##############################################################################

# Determination of the first quartile (q1)
def first_quartile(a):
    
    # This array is filled with values ​​that are less than the median of the input array
    first_half_array = [] 
    
    # Determine the median of the original matrix
    median_a = np.median(a)
    
    # Fill the first_half_array with the values
    # from the original array whose values ​​are less than the median of the original array
    for  i in a:
        if i < median_a: 
            first_half_array.append(i)
    # Determine the first quartile (here we use the numpy library)          
    q1 = np.median(first_half_array)
    
    # Returns the first quartile
    return q1 


# Determination of the third quartile (q3) of the array passed by the variable a
def third_quartile(a):
    # EThis array is filled with values ​​greater than the median of the original array passed in a
    second_half_array = [] 
    
    # Determines the median of the array passed by the variable a
    median_a = np.median(a)
    
    # Fills sencon_half_array (second half of the array) with the values ​​from the original array whose
    # values ​​are greater than the median of the original array passed in a
    for  i in a:
        if i > median_a: 
            second_half_array.append(i)
    
    # Determine the third quartile (we still use numpy in this part)          
    q3 = np.median(second_half_array)
    
    # Returns the third quartile
    return q3 

# Determines the interquartile range (IQR)
def IQR_calc(a):
                   
    q1 = first_quartile(a)
        
    q3 = third_quartile(a)
        
    IQR = q3 - q1
        
    return IQR   

print("\n ========================================================================")
print("\n The values ​​below were obtained according to the method used in statology.org")
print("\n ========================================================================")

print("\n Mixture 1")
print("\n Max Value for Mixture 1: ", max(data["Mixture 1"]))
print("\n Min VAlue for Mixture 1: ", min(data["Mixture 1"]))
print("\n 1st quartile (q1) for Mixture 1: ", first_quartile(data["Mixture 1"]))
print("\n 3rd quartile (q3) for Mixture 1: ", third_quartile(data["Mixture 1"]))
print("\n IQR_Numpy for Mixture for Mixture 1: ", IQR_calc(data["Mixture 1"]))

print("\n Mixture 2")
print("\n Max Value for Mixture 2: ", max(data["Mixture 2"]))
print("\n Min VAlue for Mixture 2: ", min(data["Mixture 2"]))
print("\n 1st quartile (q1) for Mixture 2: ", first_quartile(data["Mixture 2"]))
print("\n 3rd quartile (q3) for Mixture 2: ", third_quartile(data["Mixture 2"]))
print("\n IQR_Numpy for Mixture for Mixture 2: ", IQR_calc(data["Mixture 2"]))

print("\n Mixture 3")
print("\n Max Value for Mixture 3: ", max(data["Mixture 3"]))
print("\n Min VAlue for Mixture 3: ", min(data["Mixture 3"]))
print("\n 1st quartile (q1) for Mixture 3: ", first_quartile(data["Mixture 3"]))
print("\n 3rd quartile (q3) for Mixture 3: ", third_quartile(data["Mixture 3"]))
print("\n IQR_Numpy for Mixture for Mixture 3: ", IQR_calc(data["Mixture 3"]))

