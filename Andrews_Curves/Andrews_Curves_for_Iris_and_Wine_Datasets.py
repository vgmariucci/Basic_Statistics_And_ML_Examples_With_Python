import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import andrews_curves
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler

# Example 1: Andrews Curves for the Iris Dataset
print("=" * 60)
print("Andrews Curves for the Iris Dataset")
print("=" * 60)

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Create Andrews curves plot
plt.figure(figsize=(12, 6))
andrews_curves(iris_df, 'species', alpha = 0.4, colormap='viridis')
plt.title("Andrews Curves for the Iris Dataset", fontsize=14, fontweight='bold')
plt.xlabel("t")
plt.ylabel("Andrews Function f(t)")
plt.legend(loc = 'best')
plt.grid(True, alpha=0.3)
plt.show()

# Example 2: Andrews Curves for the Wine Dataset
print("=" * 60)
print("Andrews Curves for the Wine Dataset")
print("=" * 60)

# Load the Wine dataset
wine = load_wine()
wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)

# Standardize the features (important for Andrews curves)
scaler = StandardScaler()
wine_df_scaled = pd.DataFrame(
    scaler.fit_transform(wine_df), 
    columns=wine.feature_names
)
wine_df_scaled['wine_type'] = pd.Categorical.from_codes(wine.target, wine.target_names)

plt.figure(figsize=(12, 6))
andrews_curves(wine_df_scaled, 'wine_type', alpha = 0.3, colormap='plasma')
plt.title("Andrews Curves for the Wine Dataset", fontsize=14, fontweight='bold')
plt.xlabel("t")
plt.ylabel("Andrews Function f(t)")
plt.legend(loc = 'best')
plt.grid(True, alpha=0.3)
plt.show()