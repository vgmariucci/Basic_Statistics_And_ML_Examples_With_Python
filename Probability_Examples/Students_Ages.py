############################################################################################
# In a certain high school, some students were randomly selected
# to find out their age. Below is a sample of the ages of these students.
#
# Calculate the mean and standard deviation, and draw the boxplot, histogram, and normal 
# distribution curve graphs in one figure.
############################################################################################

# Importing the libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm

# Creating the dataframe or dataset
data = {
    'Ages': [14, 17, 18, 15, 15, 16, 17, 15, 16, 16, 15, 17, 15, 16,
             16, 18, 18, 19, 17, 16, 17, 15, 16, 17, 17, 19, 20, 18,
             17, 16, 15, 16, 16, 17, 18, 18, 17, 17, 15, 16, 16, 15]
}

# Calculating the mean and standard deviation
mean_age = np.mean(data['Ages'])
std_dev_age = np.std(data['Ages'], ddof=1)

# Creating the normal distribution curve's domain
domain = np.linspace(np.min(data['Ages']), np.max(data['Ages']), 100)

# Creating a figure with 3 subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))  # 1 row, 3 columns

# Plot 1: Histogram
axes[0].hist(data['Ages'], bins=10, color='skyblue', edgecolor='black')
axes[0].set_title('Histogram of Ages')
axes[0].set_xlabel('Ages')
axes[0].set_ylabel('Frequency')

# Plot 2: Normal Distribution Curve
axes[1].plot(domain, norm.pdf(domain, mean_age, std_dev_age), color='red', linewidth=2)
axes[1].set_title('Normal Distribution Curve')
axes[1].set_xlabel('Ages')
axes[1].set_ylabel('Probability Density')

# Plot 3: Boxplot
axes[2].boxplot(data['Ages'], patch_artist=True, boxprops=dict(facecolor='lightgreen'))
axes[2].set_title('Boxplot of Ages')
axes[2].set_ylabel('Ages')

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the figure
plt.show()

# Printing the statistics
dataframe = pd.DataFrame(data=data)
print("\nMean: ", dataframe['Ages'].mean())
print("Median: ", dataframe['Ages'].median())
print("Mode: ", dataframe['Ages'].mode()[0])
print("Sample Standard Deviation: ", dataframe['Ages'].std())
