import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Load the data
data = pd.read_csv("combined.csv")

# Ensure the relevant columns are integers
data['label'] = data['label'].astype(int)
data['transactional'] = data['transactional'].astype(int)
data['source'] = data['source'].astype(int)

# Create a 3D cube (label x transactional x source)
cube = np.zeros((3, 3, 3), dtype=int)

for label in range(3):
    for transactional in range(3):
        for source in range(3):
            # Count occurrences of each combination
            count = data[
                (data['label'] == label) &
                (data['transactional'] == transactional) &
                (data['source'] == source)
            ].shape[0]
            cube[label, transactional, source] = count

# Prepare coordinates and prevalence for plotting
x, y, z = np.indices(cube.shape)
x, y, z = x.flatten(), y.flatten(), z.flatten()
prevalence = cube.flatten()

# Filter out zero values for better visualization
nonzero_mask = prevalence > 0
x, y, z, prevalence = x[nonzero_mask], y[nonzero_mask], z[nonzero_mask], prevalence[nonzero_mask]

# Compute sizes proportional to cube root of prevalence
sizes = np.sqrt(prevalence) * 150  # Scale factor to adjust visual size

# Plot using scatter with logarithmic color scaling
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(
    x, y, z,
    c=prevalence, cmap='Blues', s=sizes, alpha=0.8, edgecolors='k',
    norm=LogNorm(vmin=prevalence.min(), vmax=prevalence.max())  # Logarithmic color scale
)

# Set axis labels
ax.set_xlabel('Label')
ax.set_ylabel('Transactional')
ax.set_zlabel('Source')

# Add a color bar with logarithmic scaling
cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label('Prevalence (Log Scale)')

plt.title('3D Scatter Plot with Logarithmic Color Scaling')
plt.show()
