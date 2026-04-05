import numpy as np
import rerun as rr
from rerun import Points3D

# Load positions and colors
positions = np.load("pointcloud_points.npy")  # shape: (N,3)
colors = np.load("pointcloud_colors.npy")  # shape: (N,3)

# Normalize colors to [0,1] if they are in 0-255
if colors.max() > 1.0:
    colors = colors / 255.0

# Start rerun session
rr.init("pointcloud_example", spawn=True)

# Log points with colors
rr.log("points", Points3D(positions=positions, colors=colors))

# Finish session
rr.shutdown()
