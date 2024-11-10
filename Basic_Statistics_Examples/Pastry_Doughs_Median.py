#################################################################################################################
#
# Calculation of the Median for the densities of the pastry doughs
#
#############################################################################################################
import numpy as np


#Creating the dataframe
data = {"Mixture 1":[22.02   ,   23.83  ,   26.67  ,    25.38  ,    25.49   ,   23.50  ,    25.90   ,  24.89],
         "Mixture 2":[21.49   ,   22.67  ,    24.62 ,     24.18 ,     22.78  ,    22.56 ,     24.46  ,  23.79],
         "Mixture 3":[20.33   ,   21.67  ,    24.67 ,     22.45 ,     22.29  ,    21.95 ,     20.49  ,   21.81]
         }


print ("\n Median for the Mixture 1: ", np.median(data["Mixture 1"]))
print ("\n Median for the Mixture 2: ", np.median(data["Mixture 2"]))
print ("\n Median for the Mixture 2: ", np.median(data["Mixture 3"]))