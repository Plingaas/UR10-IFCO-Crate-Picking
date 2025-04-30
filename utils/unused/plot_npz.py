import numpy as np
import matplotlib.pyplot as plt

# Load the .npz file
data = np.load("nndata/data(onpallet)347_1.npz")
arr = data[data.files[0]]  # assumes only one array inside

# Extract force and torque
force = arr[:, 9:12]
torque = arr[:, 12:15]

# Labels
force_labels = ["Fx", "Fy", "Fz"]
torque_labels = ["Tx", "Ty", "Tz"]

# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Force subplot
for i in range(3):
    axs[0].plot(force[:, i], label=force_labels[i])
axs[0].set_title("Force over Time")
axs[0].set_ylabel("Force (N)")
axs[0].legend()
axs[0].grid(True)

# Torque subplot
for i in range(3):
    axs[1].plot(torque[:, i], label=torque_labels[i])
axs[1].set_title("Torque over Time")
axs[1].set_xlabel("Sample Index")
axs[1].set_ylabel("Torque (Nm)")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
