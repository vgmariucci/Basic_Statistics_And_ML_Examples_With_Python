#######################################################################
# Andrews Curves and Fourier Transform
# Demonstrates how to create Andrews curves and apply Fourier Transform
# to visualize frequency components of the data.
#######################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import andrews_curves
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy import fft
from scipy.signal import find_peaks

# =============================================================================
# Helper Functions
# =============================================================================

def compute_andrews_curve(row, t_values):
    """
    Compute Andrews curve for a single data point.
    
    f(t) = x1/√2 + x2·sin(t) + x3·cos(t) + x4·sin(2t) + x5·cos(2t) + ...
    """
    n_features = len(row)
    curve = np.ones_like(t_values) * (row[0] / np.sqrt(2))
    
    for i in range(1, n_features):
        if i % 2 == 1:  # Odd index: sin
            curve += row[i] * np.sin(((i + 1) // 2) * t_values)
        else:  # Even index: cos
            curve += row[i] * np.cos((i // 2) * t_values)
    
    return curve

def extract_fourier_features(data, n_samples=200, n_coefficients=10):
    """
    Extract Fourier coefficients from Andrews curves.
    Returns both magnitude and phase information.
    """
    t_values = np.linspace(-np.pi, np.pi, n_samples)
    fourier_features = []
    
    for idx in range(len(data)):
        curve = compute_andrews_curve(data.iloc[idx].values, t_values)
        
        # Compute FFT
        fft_vals = fft.fft(curve)
        
        # Extract magnitude and phase of top coefficients
        magnitudes = np.abs(fft_vals[:n_coefficients])
        phases = np.angle(fft_vals[:n_coefficients])
        
        # Combine magnitude and phase
        features = np.concatenate([magnitudes, phases])
        fourier_features.append(features)
    
    return np.array(fourier_features)

def filter_andrews_curve(row, t_values, cutoff_freq=5):
    """
    Apply low-pass filter to Andrews curve using FFT.
    """
    curve = compute_andrews_curve(row, t_values)
    
    # FFT
    fft_vals = fft.fft(curve)
    
    # Apply low-pass filter
    fft_filtered = fft_vals.copy()
    fft_filtered[cutoff_freq:-cutoff_freq] = 0
    
    # Inverse FFT
    filtered_curve = fft.ifft(fft_filtered).real
    
    return curve, filtered_curve

# =============================================================================
# Example 1: Fourier Spectrum Analysis
# =============================================================================
print("=" * 70)
print("Example 1: Fourier Spectrum Analysis of Andrews Curves")
print("=" * 70)

# Load and prepare Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
scaler = StandardScaler()
iris_scaled = pd.DataFrame(
    scaler.fit_transform(iris_df),
    columns=iris_df.columns
)
iris_scaled['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Compute Andrews curves and their FFT
t_values = np.linspace(-np.pi, np.pi, 500)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

species_colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}

for species in iris.target_names:
    species_data = iris_scaled[iris_scaled['species'] == species].iloc[:, :-1]
    color = species_colors[species]
    
    # Plot a few example curves
    for idx in range(min(10, len(species_data))):
        curve = compute_andrews_curve(species_data.iloc[idx], t_values)
        axes[0, 0].plot(t_values, curve, color=color, alpha=0.3, linewidth=0.8)
        
        # Compute FFT
        fft_vals = fft.fft(curve)
        freqs = fft.fftfreq(len(curve), d=(2*np.pi/len(curve)))
        
        # Plot magnitude spectrum (only positive frequencies)
        n = len(freqs) // 2
        axes[0, 1].plot(freqs[:n], np.abs(fft_vals[:n]), 
                       color=color, alpha=0.3, linewidth=0.8)

# Add legend entries
for species, color in species_colors.items():
    axes[0, 0].plot([], [], color=color, label=species, linewidth=2)
    axes[0, 1].plot([], [], color=color, label=species, linewidth=2)

axes[0, 0].set_title('Andrews Curves (Time Domain)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('t')
axes[0, 0].set_ylabel('f(t)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].set_title('Fourier Spectrum (Frequency Domain)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Frequency')
axes[0, 1].set_ylabel('Magnitude')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim(0, 50)

# Average spectrum per species
for species in iris.target_names:
    species_data = iris_scaled[iris_scaled['species'] == species].iloc[:, :-1]
    color = species_colors[species]
    
    all_spectra = []
    for idx in range(len(species_data)):
        curve = compute_andrews_curve(species_data.iloc[idx], t_values)
        fft_vals = fft.fft(curve)
        all_spectra.append(np.abs(fft_vals))
    
    avg_spectrum = np.mean(all_spectra, axis=0)
    freqs = fft.fftfreq(len(avg_spectrum), d=(2*np.pi/len(curve)))
    n = len(freqs) // 2
    
    axes[1, 0].plot(freqs[:n], avg_spectrum[:n], color=color, 
                   label=species, linewidth=2)

axes[1, 0].set_title('Average Fourier Spectrum per Species', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Frequency')
axes[1, 0].set_ylabel('Average Magnitude')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim(0, 50)

# Dominant frequencies analysis
dominant_freqs = {species: [] for species in iris.target_names}

for species in iris.target_names:
    species_data = iris_scaled[iris_scaled['species'] == species].iloc[:, :-1]
    
    for idx in range(len(species_data)):
        curve = compute_andrews_curve(species_data.iloc[idx], t_values)
        fft_vals = fft.fft(curve)
        spectrum = np.abs(fft_vals[:len(fft_vals)//2])
        
        # Find peaks in spectrum
        peaks, _ = find_peaks(spectrum, height=0.5)
        if len(peaks) > 0:
            dominant_freqs[species].extend(peaks[:3])  # Top 3 peaks

# Plot histogram of dominant frequencies
for species, color in species_colors.items():
    if len(dominant_freqs[species]) > 0:
        axes[1, 1].hist(dominant_freqs[species], bins=20, alpha=0.5, 
                       color=color, label=species, edgecolor='black')

axes[1, 1].set_title('Distribution of Dominant Frequencies', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Frequency Component')
axes[1, 1].set_ylabel('Count')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# =============================================================================
# Example 2: Low-Pass Filtering (Noise Reduction)
# =============================================================================
print("=" * 70)
print("Example 2: Low-Pass Filtering for Noise Reduction")
print("=" * 70)

# Add noise to data
np.random.seed(42)
iris_noisy = iris_scaled.iloc[:, :-1].copy()
noise = np.random.normal(0, 0.3, iris_noisy.shape)
iris_noisy = iris_noisy + noise

t_values = np.linspace(-np.pi, np.pi, 500)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Show examples for each species
for idx, species in enumerate(iris.target_names):
    species_idx = np.where(iris.target == idx)[0][0]
    
    # Original (clean) curve
    clean_curve = compute_andrews_curve(iris_scaled.iloc[species_idx, :-1], t_values)
    
    # Noisy curve
    noisy_curve = compute_andrews_curve(iris_noisy.iloc[species_idx], t_values)
    
    # Filtered curve
    _, filtered_curve = filter_andrews_curve(iris_noisy.iloc[species_idx], t_values, cutoff_freq=10)
    
    # Plot curves
    axes[0, idx].plot(t_values, clean_curve, 'g-', linewidth=2, label='Original', alpha=0.7)
    axes[0, idx].plot(t_values, noisy_curve, 'r--', linewidth=1, label='Noisy', alpha=0.5)
    axes[0, idx].set_title(f'{species.capitalize()} - Time Domain', fontweight='bold')
    axes[0, idx].legend()
    axes[0, idx].grid(True, alpha=0.3)
    
    axes[1, idx].plot(t_values, noisy_curve, 'r--', linewidth=1, label='Noisy', alpha=0.5)
    axes[1, idx].plot(t_values, filtered_curve, 'b-', linewidth=2, label='Filtered', alpha=0.7)
    axes[1, idx].set_title(f'{species.capitalize()} - Filtered', fontweight='bold')
    axes[1, idx].set_xlabel('t')
    axes[1, idx].legend()
    axes[1, idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# Example 3: Feature Extraction with Fourier Coefficients
# =============================================================================
print("=" * 70)
print("Example 3: Classification Using Fourier Features")
print("=" * 70)

# Load wine dataset for more challenging classification
wine = load_wine()
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
scaler = StandardScaler()
wine_scaled = pd.DataFrame(
    scaler.fit_transform(wine_df),
    columns=wine_df.columns
)

# Extract Fourier features
print("Extracting Fourier features from Andrews curves...")
fourier_feats = extract_fourier_features(wine_scaled, n_samples=200, n_coefficients=15)
print(f"  Original features: {wine_scaled.shape[1]}")
print(f"  Fourier features: {fourier_feats.shape[1]} (15 magnitudes + 15 phases)")

# Split data
X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    wine_scaled, wine.target, test_size=0.3, random_state=42, stratify=wine.target
)

X_train_fourier, X_test_fourier = train_test_split(
    fourier_feats, test_size=0.3, random_state=42, stratify=wine.target
)

# Train classifiers
print("\nTraining Random Forest classifiers...")

# Original features
rf_orig = RandomForestClassifier(n_estimators=100, random_state=42)
rf_orig.fit(X_train_orig, y_train)
y_pred_orig = rf_orig.predict(X_test_orig)
acc_orig = accuracy_score(y_test, y_pred_orig)

# Fourier features
rf_fourier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_fourier.fit(X_train_fourier, y_train)
y_pred_fourier = rf_fourier.predict(X_test_fourier)
acc_fourier = accuracy_score(y_test, y_pred_fourier)

# Combined features
X_train_combined = np.hstack([X_train_orig, X_train_fourier])
X_test_combined = np.hstack([X_test_orig, X_test_fourier])

rf_combined = RandomForestClassifier(n_estimators=100, random_state=42)
rf_combined.fit(X_train_combined, y_train)
y_pred_combined = rf_combined.predict(X_test_combined)
acc_combined = accuracy_score(y_test, y_pred_combined)

print("\n" + "=" * 60)
print("CLASSIFICATION RESULTS")
print("=" * 60)
print(f"Original Features:          {acc_orig:.4f} accuracy")
print(f"Fourier Features Only:      {acc_fourier:.4f} accuracy")
print(f"Combined Features:          {acc_combined:.4f} accuracy")
print("=" * 60)

# Visualize results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

results = [acc_orig, acc_fourier, acc_combined]
labels = ['Original\nFeatures', 'Fourier\nFeatures', 'Combined\nFeatures']
colors = ['steelblue', 'coral', 'mediumseagreen']

bars = axes[0].bar(labels, results, color=colors, edgecolor='black', linewidth=2)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Classification Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylim([0.85, 1.0])
axes[0].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, results):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

# Feature importance for combined model
importances = rf_combined.feature_importances_
n_orig = X_train_orig.shape[1]
orig_importance = importances[:n_orig].sum()
fourier_importance = importances[n_orig:].sum()

axes[1].pie([orig_importance, fourier_importance], 
           labels=['Original Features', 'Fourier Features'],
           colors=['steelblue', 'coral'],
           autopct='%1.1f%%',
           startangle=90,
           textprops={'fontsize': 11, 'fontweight': 'bold'})
axes[1].set_title('Feature Importance Distribution\n(Combined Model)', 
                 fontsize=12, fontweight='bold')

# Top Fourier frequencies
top_fourier_idx = np.argsort(importances[n_orig:])[-10:][::-1]
top_fourier_imp = importances[n_orig:][top_fourier_idx]

axes[2].barh(range(len(top_fourier_idx)), top_fourier_imp, color='coral', edgecolor='black')
axes[2].set_yticks(range(len(top_fourier_idx)))
axes[2].set_yticklabels([f'Coef {i}' for i in top_fourier_idx])
axes[2].set_xlabel('Importance', fontsize=11)
axes[2].set_title('Top 10 Fourier Feature Importances', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='x')
axes[2].invert_yaxis()

plt.tight_layout()
plt.show()

# =============================================================================
# Example 4: Dimensionality Reduction via Frequency Selection
# =============================================================================
print("=" * 70)
print("Example 4: Dimensionality Reduction with Frequency Selection")
print("=" * 70)

# Use different numbers of Fourier coefficients
n_coef_range = [3, 5, 10, 15, 20, 25, 30]
accuracies = []

for n_coef in n_coef_range:
    fourier_feats_temp = extract_fourier_features(wine_scaled, n_samples=200, n_coefficients=n_coef)
    X_train_temp, X_test_temp = train_test_split(
        fourier_feats_temp, test_size=0.3, random_state=42, stratify=wine.target
    )
    
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_temp.fit(X_train_temp, y_train)
    y_pred_temp = rf_temp.predict(X_test_temp)
    acc_temp = accuracy_score(y_test, y_pred_temp)
    accuracies.append(acc_temp)
    print(f"  {n_coef:2d} coefficients: {acc_temp:.4f} accuracy ({n_coef*2} features)")

# Plot accuracy vs number of coefficients
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.plot(n_coef_range, accuracies, 'o-', linewidth=2, markersize=8, 
        color='darkblue', label='Fourier Features')
ax.axhline(y=acc_orig, color='red', linestyle='--', linewidth=2, 
          label=f'Original Features (baseline: {acc_orig:.3f})')
ax.set_xlabel('Number of Fourier Coefficients', fontsize=12)
ax.set_ylabel('Classification Accuracy', fontsize=12)
ax.set_title('Accuracy vs Number of Fourier Coefficients', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

# Annotate best point
best_idx = np.argmax(accuracies)
ax.annotate(f'Best: {n_coef_range[best_idx]} coefs\n{accuracies[best_idx]:.3f}',
           xy=(n_coef_range[best_idx], accuracies[best_idx]),
           xytext=(n_coef_range[best_idx]+5, accuracies[best_idx]-0.02),
           arrowprops=dict(arrowstyle='->', color='red', lw=2),
           fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

# =============================================================================
# Example 5: Curve Similarity in Frequency Domain
# =============================================================================
print("=" * 70)
print("Example 5: Computing Curve Similarity in Frequency Domain")
print("=" * 70)

# Select samples from each wine type
wine_samples = []
wine_labels = []
for wine_type in range(3):
    idx = np.where(wine.target == wine_type)[0][0]
    wine_samples.append(wine_scaled.iloc[idx].values)
    wine_labels.append(wine.target_names[wine_type])

t_values = np.linspace(-np.pi, np.pi, 300)

# Compute curves and their FFTs
curves = []
ffts = []

for sample in wine_samples:
    curve = compute_andrews_curve(sample, t_values)
    curves.append(curve)
    fft_vals = np.abs(fft.fft(curve))
    ffts.append(fft_vals)

# Compute pairwise distances
from scipy.spatial.distance import euclidean

print("\nPairwise Euclidean Distances:")
print("\nTime Domain (Andrews Curves):")
for i in range(len(curves)):
    for j in range(i+1, len(curves)):
        dist = euclidean(curves[i], curves[j])
        print(f"  {wine_labels[i]:12s} vs {wine_labels[j]:12s}: {dist:.4f}")

print("\nFrequency Domain (Fourier Spectra):")
for i in range(len(ffts)):
    for j in range(i+1, len(ffts)):
        dist = euclidean(ffts[i][:50], ffts[j][:50])  # Compare first 50 frequencies
        print(f"  {wine_labels[i]:12s} vs {wine_labels[j]:12s}: {dist:.4f}")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

colors = ['red', 'green', 'blue']

# Plot curves
for curve, label, color in zip(curves, wine_labels, colors):
    axes[0, 0].plot(t_values, curve, color=color, label=label, linewidth=2)

axes[0, 0].set_title('Andrews Curves (Time Domain)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('t')
axes[0, 0].set_ylabel('f(t)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot FFT magnitudes
freqs = fft.fftfreq(len(curves[0]), d=(2*np.pi/len(curves[0])))
n = len(freqs) // 2

for fft_vals, label, color in zip(ffts, wine_labels, colors):
    axes[0, 1].plot(freqs[:n], fft_vals[:n], color=color, label=label, linewidth=2)

axes[0, 1].set_title('Fourier Spectra (Frequency Domain)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Frequency')
axes[0, 1].set_ylabel('Magnitude')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim(0, 40)

# Distance matrices
time_distances = np.zeros((3, 3))
freq_distances = np.zeros((3, 3))

for i in range(3):
    for j in range(3):
        time_distances[i, j] = euclidean(curves[i], curves[j])
        freq_distances[i, j] = euclidean(ffts[i][:50], ffts[j][:50])

# Heatmap for time domain
im1 = axes[1, 0].imshow(time_distances, cmap='YlOrRd', aspect='auto')
axes[1, 0].set_xticks(range(3))
axes[1, 0].set_yticks(range(3))
axes[1, 0].set_xticklabels(wine_labels, rotation=45)
axes[1, 0].set_yticklabels(wine_labels)
axes[1, 0].set_title('Distance Matrix - Time Domain', fontsize=12, fontweight='bold')
plt.colorbar(im1, ax=axes[1, 0])

# Add text annotations
for i in range(3):
    for j in range(3):
        text = axes[1, 0].text(j, i, f'{time_distances[i, j]:.2f}',
                              ha="center", va="center", color="black", fontweight='bold')

# Heatmap for frequency domain
im2 = axes[1, 1].imshow(freq_distances, cmap='YlOrRd', aspect='auto')
axes[1, 1].set_xticks(range(3))
axes[1, 1].set_yticks(range(3))
axes[1, 1].set_xticklabels(wine_labels, rotation=45)
axes[1, 1].set_yticklabels(wine_labels)
axes[1, 1].set_title('Distance Matrix - Frequency Domain', fontsize=12, fontweight='bold')
plt.colorbar(im2, ax=axes[1, 1])

# Add text annotations
for i in range(3):
    for j in range(3):
        text = axes[1, 1].text(j, i, f'{freq_distances[i, j]:.2f}',
                              ha="center", va="center", color="black", fontweight='bold')

plt.tight_layout()
plt.show()