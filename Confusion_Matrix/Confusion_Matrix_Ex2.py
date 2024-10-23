# Import the necessary libraries for plotting and numerical operations
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

# Generate 1000 random binary values (1 or 0) as the 'actual' outcomes 
# with a 90% chance of being 1 (success) and 10% chance of being 0 (failure)
actual = np.random.binomial(1, 0.9, size=1000)

# Generate 1000 random binary values as the 'predicted' outcomes
# using the same probability distribution as the 'actual' outcomes
predicted = np.random.binomial(1, 0.9, size=1000)

# Create a confusion matrix comparing 'actual' and 'predicted' values
confusion_matrix = metrics.confusion_matrix(actual, predicted)

# Display the confusion matrix as a visual plot
# Set display labels for True (1) and False (0)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[True, False])

# Plot and display the confusion matrix
cm_display.plot()
plt.show()
