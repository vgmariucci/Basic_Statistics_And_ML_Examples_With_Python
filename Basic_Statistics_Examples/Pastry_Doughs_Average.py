#######################################################################################################################################

# Calculation of the Arithmetic Mean for the densities of the pastry masses:

#######################################################################################################################################

import numpy as np

#Creating the dataframe
pastry_densities_data = {
        "Mixture 1":[22.02   ,   23.83  ,   26.67  ,    25.38  ,    25.49   ,   23.50  ,    25.90   ,  24.89],
        "Mixture 2":[21.49   ,   22.67  ,    24.62 ,     24.18 ,     22.78  ,    22.56 ,     24.46  ,  23.79],
        "Mixture 3":[20.33   ,   21.67  ,    24.67 ,     22.45 ,     22.29  ,    21.95 ,     20.49  ,   21.81]
        }

# Calculating the average densities for each mixture
for key in pastry_densities_data:
    pastry_densities_data[key] = np.mean(pastry_densities_data[key])
    print (f"\n Average density for the {key}: {pastry_densities_data[key]}")


