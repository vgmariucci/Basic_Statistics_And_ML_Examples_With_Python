##############################################################
# Andrews Curves with Custom Styling: 
# Demonstrates how to create Andrews curves, adjust smoothness,
# and apply custom styling to the plot.
##############################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import andrews_curves
from sklearn.datasets import load_iris

# Example for customizing Andrews Curves
print("=" * 60)
print("Andrews Curves with Custom Styling Example")
print("=" * 60)

# Sample from sklearn's iris dataset for cleanner visualization
# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
iris_sample = iris_df.sample(n = 50, random_state= 42)

fig, axes = plt.subplots(1, 2, figsize=(10, 6))

# Plot 1: With more samples
andrews_curves(
    iris_sample, 
    'species', 
    ax=axes[0],
    colormap='tab10',
    samples = 500, 
    alpha=0.6
    )

axes[0].set_title("More Sample Points (samples = 500)", fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Plot 2: With fewer samples and custom styling
andrews_curves(
    iris_sample, 
    'species', 
    ax=axes[1], 
    alpha=0.6, 
    colormap='tab10',
    samples=5
    )

axes[1].set_title("Fewer Sample Points (samples = 5)", fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()