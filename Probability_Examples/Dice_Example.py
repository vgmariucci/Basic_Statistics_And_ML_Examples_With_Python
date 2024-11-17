###############################################################################
#  Script used to check whether a dice is biased or not
############################################################################

# Importing the libraries
import random
import pandas as pd
from matplotlib import pyplot as plt

# Define the sizes of the lists
list_sizes = [20, 200, 2000, 20000, 200000, 2000000]

# Create a figure with subplots to plot all histograms in one frame
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))  # 2 rows, 3 columns

# Loop through each list size
for idx, size in enumerate(list_sizes):
    # Create a list of random numbers (1-6) for the given size
    list_of_numbers_6_sided_dice = {'Sides': [random.randint(1, 6) for _ in range(size)]}
    
    # Pass the list to a dataframe
    df = pd.DataFrame(data=list_of_numbers_6_sided_dice, columns=['Sides'])
    
    # Plot the histogram in the corresponding subplot
    ax = axes[idx // 3, idx % 3]  # Determine subplot position
    df.plot.hist(
        ax=ax, 
        align='right', 
        rwidth=0.9, 
        bins=6, 
        legend=False,
        color='skyblue'
    )
    ax.set_title(f"Histogram for {size} Rolls")
    ax.set_xlabel("Dice Sides")
    ax.set_ylabel("Frequency")

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the histograms
plt.show()

