############################################################################################################
#
# Construction of histograms for the densities of the pastry masses.
#
# Construction of frequency histogram and relative frequency histogram
############################################################################################################

from fractions import Fraction
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Creating the dataframe
dados = {
        "Mixture 1":[22.02   ,   23.83  ,   26.67  ,    25.38  ,    25.49   ,   23.50  ,    25.90   ,  24.89],
         "Mixture 2":[21.49   ,   22.67  ,    24.62 ,     24.18 ,     22.78  ,    22.56 ,     24.46  ,  23.79],
         "Mixture 3":[20.33   ,   21.67  ,    24.67 ,     22.45 ,     22.29  ,    21.95 ,     20.49  ,   21.81]
         }



#Construction of the Frequency Histogram for the dataframe of the pastel mass densities

# Defining the size of the figure that will contain the graphs (histograms)
fig_Frequency_Histogram = plt.figure(figsize=(10, 20))

Frequency_Histogram_Mixture_1 = fig_Frequency_Histogram.add_subplot(511)

Frequency_Histogram_Mixture_1.set_title("Frequency Histogram for Mixture 1")

Frequency_Histogram_Mixture_1.hist(dados["Mixture 1"], bins = 50, color = "blue", edgecolor = "black", lw =1)


Frequency_Histogram_Mixture_2 = fig_Frequency_Histogram.add_subplot(513)

Frequency_Histogram_Mixture_2.set_title("Frequency Histogram for Mixture 2")

Frequency_Histogram_Mixture_2.hist(dados["Mixture 2"], bins = 50, color = "blue", edgecolor = "black", lw =1)


Frequency_Histogram_Mixture_3 = fig_Frequency_Histogram.add_subplot(515)

Frequency_Histogram_Mixture_3.set_title("Frequency Histogram for Mixture 3")

Frequency_Histogram_Mixture_3.hist(dados["Mixture 3"], bins = 50, color = "blue", edgecolor = "black", lw =1)




#Construction of relative frequency histograms for the densities of the pastel masses

# Definition of the size of the figure that will contain the relative frequency histograms
fig_Relative_Frequency_Histogram = plt.figure(figsize=(10, 20))

Relative_Frequency_Histogram_Mixture_1 = fig_Relative_Frequency_Histogram.add_subplot(511)

Relative_Frequency_Histogram_Mixture_1.set_title("Relative Frequency Histogram for the Mixture 1")

Relative_Frequency_Histogram_Mixture_1.hist(dados["Mixture 1"], weights = np.ones_like(dados["Mixture 1"]) / len(dados["Mixture 1"]), bins = 50, color = "blue", edgecolor = "black", lw =1)


Relative_Frequency_Histogram_Mixture_2 = fig_Relative_Frequency_Histogram.add_subplot(513)

Relative_Frequency_Histogram_Mixture_2.set_title("Relative Frequency Histogram for the Mixture 2")

Relative_Frequency_Histogram_Mixture_2.hist(dados["Mixture 2"], weights = np.ones_like(dados["Mixture 2"]) / len(dados["Mixture 2"]), bins = 50, color = "blue", edgecolor = "black", lw =1)


Relative_Frequency_Histogram_Mixture_3 = fig_Relative_Frequency_Histogram.add_subplot(515)

Relative_Frequency_Histogram_Mixture_3.set_title("Relative Frequency Histogram for the Mixture 3")

Relative_Frequency_Histogram_Mixture_3.hist(dados["Mixture 3"], weights = np.ones_like(dados["Mixture 3"]) / len(dados["Mixture 3"]), bins = 50, color = "blue", edgecolor = "black", lw =1)


plt.show()