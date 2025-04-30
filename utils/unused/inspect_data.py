import numpy as np

# Load the file
data = np.load("data.npz")

# Check keys (in case you saved multiple arrays)
print(data.files)  # Example output: ['measurements']

# Access the saved array
measurements = data["measurements"]

# Print shape and preview
print(measurements.shape)      # e.g. (100, 12)
print(measurements[:5])        # show first 5 samples
