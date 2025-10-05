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
pastry_densities_data = {
    'Mixture 1':[22.02   ,   23.83  ,   26.67  ,    25.38  ,    25.49   ,   23.50  ,    25.90   ,  24.89],
    'Mixture 2':[21.49   ,   22.67  ,    24.62 ,     24.18 ,     22.78  ,    22.56 ,     24.46  ,  23.79],
    'Mixture 3':[20.33   ,   21.67  ,    24.67 ,     22.45 ,     22.29  ,    21.95 ,     20.49  ,   21.81]
    }

# Create a DataFrame
df = pd.DataFrame(data = pastry_densities_data)

# View dataframe
print(df)

# boxplot for dataframe = df
boxplot = df[['Mixture 1', 'Mixture 2', 'Mixture 3']].plot(kind='box', title='boxplot')
plt.show()

# Calculating the first and third quartiles with numpy resources only
# and the interquartile range (IQR)
print("\n" + 50*"=")
print("\n The values ​​below were obtained using numpy")
print("\n" + 50*"=")

for key in pastry_densities_data:

    q3_Numpy_Mixture1, q1_Numpy_Mixture1 = np.percentile(pastry_densities_data[key], [75, 25])

    # Calculating the interquartile range (IQR) with numpy
    IQR_Numpy_Mixture1 = q3_Numpy_Mixture1 - q1_Numpy_Mixture1

    print(f"\n {key}")
    print(f"\n Max Value for {key}: {max(pastry_densities_data[key])}")
    print(f"\n Min Value for {key}: {min(pastry_densities_data[key])}")
    print(f"\n 1st quartile (q1) for {key}: {q1_Numpy_Mixture1:.2f}")
    print(f"\n 3rd quartile (q3) for {key}: {q3_Numpy_Mixture1:.2f}")
    print(f"\n IQR_Numpy for {key}: {IQR_Numpy_Mixture1:.2f}")



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

# Printing the results obtained with the method used on the statology.org website
print("\n" + 70*"=")
print("\n The values ​​below were obtained according to the method used in statology.org")
print("\n" + 70*"=")

for key in pastry_densities_data:
    print(f"\n {key}")
    print(f"\n Max Value for {key}: {max(pastry_densities_data[key])}")
    print(f"\n Min Value for {key}: {min(pastry_densities_data[key])}")
    print(f"\n 1st quartile (q1) for {key}: {first_quartile(pastry_densities_data[key]):.2f}")
    print(f"\n 3rd quartile (q3) for {key}: {third_quartile(pastry_densities_data[key]):.2f}")
    print(f"\n IQR_Numpy for {key}: {IQR_calc(pastry_densities_data[key]):.2f}")


