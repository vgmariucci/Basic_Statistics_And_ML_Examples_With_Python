#####################################################################
# Andrews Curves for Outliers Detection
#####################################################################

# This script demonstrates the use of Andrews curves to visualize and detect outliers in a dataset.
# We will create a synthetic dataset with outliers and plot the Andrews curves to identify them.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import andrews_curves

print("=" * 60)
print("Andrews Curves for Outliers Detection")
print("=" * 60)

# Create a synthetic dataset with outliers
np.random.seed(42)
n_samples = 100

# Generate normal observations (two clusters)
cluster1 = np.random.randn(n_samples // 2, 4) * 0.5 + np.array([1, 2, 1, 2])
cluster2 = np.random.randn(n_samples // 2, 4) * 0.5 + np.array([5, 4, 5, 4])

# Add some outliers
outliers = np.array([
    [10, 10, 10, 10],
    [-5, -5, -5, -5],
    [8, -3, 7, -2]
])

# Combine clusters and outliers
data = np.vstack([cluster1, cluster2, outliers])
labels =  ['normal'] * n_samples + ['outlier'] * len(outliers)

custom_df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3', 'feature4'])
custom_df['label'] = labels

plt.figure(figsize=(12, 6))
andrews_curves(custom_df, 'label', alpha=0.5, color=['blue', 'red'])
plt.title("Andrews Curves for Outliers Detection", fontsize=14, fontweight='bold')
plt.xlabel("t")
plt.ylabel("Andrews Function f(t)")
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.show()

