########################################################################################################
#
# Calculation of the Standard Deviation for the densities of the pastel masses
#
########################################################################################################
import statistics as stat
import numpy as np


#Creating the dataframe
data = {"Mixture 1":[22.02   ,   23.83  ,   26.67  ,    25.38  ,    25.49   ,   23.50  ,    25.90   ,  24.89],
         "Mixture 2":[21.49   ,   22.67  ,    24.62 ,     24.18 ,     22.78  ,    22.56 ,     24.46  ,  23.79],
         "Mixture 3":[20.33   ,   21.67  ,    24.67 ,     22.45 ,     22.29  ,    21.95 ,     20.49  ,   21.81]
         }

# STD unsing numpy library
def Calc_Population_Standard_Deviation_Numpy(a):
    
    Population_STD_Numpy = np.std(a)
    
    return Population_STD_Numpy

def Calc_Sample_Standard_Deviation_Numpy(a):
    
    Sample_STD_Numpy = np.std(a, ddof = 1)
    
    return Sample_STD_Numpy


# STD unsing statistics library
def Calc_Population_Standard_Deviation_Stat(a):
    
    Population_STD_Stat = stat.pstdev(a)
    
    return Population_STD_Stat


def Calc_Sample_Standard_Deviation_Stat(a):
    
    Sample_STD_Stat = stat.stdev(a)
    
    return Sample_STD_Stat


print("\n Population STD for the Mixture 1 (obtained with numpy): ", Calc_Population_Standard_Deviation_Numpy(data["Mixture 1"]))
print("\n Population STD for the Mixture 2 (obtained with numpy): ", Calc_Population_Standard_Deviation_Numpy(data["Mixture 2"]))
print("\n Population STD for the Mixture 3 (obtained with numpy): ", Calc_Population_Standard_Deviation_Numpy(data["Mixture 3"]))

print("\n Sample STD for the Mixture 1 (obtained with numpy): ", Calc_Sample_Standard_Deviation_Numpy(data["Mixture 1"]))
print("\n Sample STD for the Mixture 2 (obtained with numpy): ", Calc_Sample_Standard_Deviation_Numpy(data["Mixture 2"]))
print("\n Sample STD for the Mixture 3 (obtained with numpy): ", Calc_Sample_Standard_Deviation_Numpy(data["Mixture 3"]))


print("\n Population STD for the Mixture 1 (obtido com statistics): ", Calc_Population_Standard_Deviation_Stat(data["Mixture 1"]))
print("\n Population STD for the Mixture 2 (obtido com statistics): ", Calc_Population_Standard_Deviation_Stat(data["Mixture 2"]))
print("\n Population STD for the Mixture 3 (obtido com statistics): ", Calc_Population_Standard_Deviation_Stat(data["Mixture 3"]))

print("\n Sample STD for the Mixture 1 (obtido com statistics): ", Calc_Sample_Standard_Deviation_Stat(data["Mixture 1"]))
print("\n Sample STD for the Mixture 2 (obtido com statistics): ", Calc_Sample_Standard_Deviation_Stat(data["Mixture 2"]))
print("\n Sample STD for the Mixture 3 (obtido com statistics): ", Calc_Sample_Standard_Deviation_Stat(data["Mixture 3"]))




