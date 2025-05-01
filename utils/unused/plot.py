import numpy as np
import matplotlib.pyplot as plt

# Load .npz file
data = np.load("nndata/data(onpallet)23_0.npz")  # Replace with your actual file name
array_key = list(data.keys())[0]  # Get the first key in the file
full_array = data[array_key]      # Shape: (N, 15)

# Last crate position
x = full_array[:, 0]
y = full_array[:, 1]
z = full_array[:, 2] + 207.5

# Robot TCP
x_tcp = full_array[:, 3]
y_tcp = full_array[:, 4]
z_tcp = full_array[:, 5]

rx_tcp = full_array[:, 6]
ry_tcp = full_array[:, 7]
rz_tcp = full_array[:, 8]

# FTS
fx = full_array[:, 9]
fy = full_array[:, 10]
fz = full_array[:, 11]

tx = full_array[:, 12]
ty = full_array[:, 13]
tz = full_array[:, 14]

labels_force = ['fx', 'fy', 'fz']
labels_torque = ['tx', 'ty', 'tz', "Z-position"]
force_limits = [(-20, 20), (-20, 20), (-100, 20)]
torque_limits = [(-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5), (0.46, 0.54)]

forces = [fx, fy, fz]
torques = [tx, ty, tz, z_tcp]

# Plot in 2x3 layout
fig, axs = plt.subplots(2, 4, figsize=(12, 6))
fig.suptitle('FAIL')

# Top row: forces
for i in range(3):
    axs[0, i].plot(forces[i])
    axs[0, i].set_title(labels_force[i])
    axs[0, i].set_ylabel('Force (N)')
    axs[0, i].set_xlabel('Time step')
    axs[0, i].set_ylim(force_limits[i])


# Bottom row: torques
for i in range(4):
    axs[1, i].plot(torques[i])
    axs[1, i].set_title(labels_torque[i])
    axs[1, i].set_ylabel('Torque (Nm)')
    axs[1, i].set_xlabel('Time step')
    axs[1, i].set_ylim(torque_limits[i])

axs[1, 3].axhline(y=0.480, color="r", linestyle="--")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
