################################################################################################################
#
# Histogram charts
#
# A histogram is a type of chart that allows us to visualize the distribution of values in a dataset.

# The x-axis displays the values in the dataset and the y-axis shows the frequency of each value.

# Depending on the values in the dataset, a histogram can take on many different shapes.

# 1. Bell-Shaped
# A histogram is bell-shaped if it resembles a “bell” curve and has one single peak in the middle of the distribution.
# The most common real-life example of this type of distribution is the normal distribution.

# 2. Uniform
# A histogram is described as “uniform” if every value in a dataset occurs roughly the same number of times. 
# This type of histogram often looks like a rectangle with no clear peaks.

# 3. Bimodal
# A histogram is described as “bimodal” if it has two distinct peaks.
# We often say that this type of distribution has multiple modes – that is, 
# multiple values occur most frequently in the dataset.

# 4. Multimodal
# A histogram is described as “multimodal” if it has more than two distinct peaks.

# 5. Left Skewed

# A histogram is left skewed if it has a “tail” on the left side of the distribution. 
# Sometimes this type of distribution is also called “negatively” skewed.

# 6. Right Skewed
# A histogram is right skewed if it has a “tail” on the right side of the distribution. 
# Sometimes this type of distribution is also called “positively” skewed.

# 7. Random
# The shape of a distribution can be described as “random” if there is no clear pattern in the data at all.

# Relative Frequency Histogram: Definition

# A close cousin of a frequency table is a relative frequency table, 
# which simply lists the frequencies of each class as a percentage of the whole.

# Similar to a frequency histogram, this type of histogram displays the classes along the x-axis of the graph 
# and uses bars to represent the relative frequencies of each class along the y-axis.

# The only difference is the labels used on the y-axis. Instead of displaying raw frequencies,
# a relative frequency histogram displays percentages.

# Note that a frequency histogram and a relative frequency histogram will both look the exact same.
# The only difference is the values displayed on the y-axis.
################################################################################################################

from fractions import Fraction
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

Lengths_of_the_Screws_From_Supplier_A = (81  ,    70  ,    83   ,   72  ,    78   ,   81   ,   81  ,    80   ,   80   ,  79)   # Array A

Lengths_of_the_Screws_From_Supplier_B = (80  ,    80  ,    80   ,   79  ,    79   ,   78   ,   81   ,   81   ,   80   ,   81)  # Array B



#Build Frequencies Histogram chart for tables of suppliers A and B

# specifying figure size
fig_Frequency_Histogram = plt.figure(figsize=(10, 5))

Frequency_Histogram_A = fig_Frequency_Histogram.add_subplot(121)

Frequency_Histogram_A.set_title("Frequency histogram for supplier A")

Frequency_Histogram_A.hist(Lengths_of_the_Screws_From_Supplier_A, bins = 50, color = "blue", edgecolor = "black", lw =1)


Frequency_Histogram_B = fig_Frequency_Histogram.add_subplot(122)

Frequency_Histogram_B.set_title("Frequency histogram for supplier B")

Frequency_Histogram_B.hist(Lengths_of_the_Screws_From_Supplier_B, bins = 50, color = "blue", edgecolor = "black", lw =1)


#Build Relative Frequency Histogram chart for tables of suppliers A and B

# specifying figure size
fig_Relative_Frequency_Histogram = plt.figure(figsize=(10, 5))

Relative_Frequency_Histogram_A = fig_Relative_Frequency_Histogram.add_subplot(121)

Relative_Frequency_Histogram_A.set_title("Relative Frequency Histogram for supplier A")

Relative_Frequency_Histogram_A.hist(Lengths_of_the_Screws_From_Supplier_A, weights = np.ones_like(Lengths_of_the_Screws_From_Supplier_A) / len(Lengths_of_the_Screws_From_Supplier_A), bins = 50, color = "blue", edgecolor = "black", lw =1)

Relative_Frequency_Histogram_B = fig_Relative_Frequency_Histogram.add_subplot(122)

Relative_Frequency_Histogram_B.set_title("Relative Frequency Histogram for supplier B")

Relative_Frequency_Histogram_B.hist(Lengths_of_the_Screws_From_Supplier_B, weights = np.ones_like(Lengths_of_the_Screws_From_Supplier_B) / len(Lengths_of_the_Screws_From_Supplier_B), bins = 50, color = "blue", edgecolor = "black", lw =1)


plt.show()