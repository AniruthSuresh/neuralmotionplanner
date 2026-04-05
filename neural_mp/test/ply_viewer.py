import matplotlib.pyplot as plt
import numpy as np

# 1. Load the data
points = np.load("pointcloud_points.npy")
colors = np.load("pointcloud_colors.npy")

print(f"Loaded {len(points)} points.")

# 2. Subsample to make rendering faster (optional, keeps 1 out of every 5 points)
step = 5
points_sub = points[::step]
colors_sub = colors[::step]

# 3. Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(points_sub[:, 0], points_sub[:, 1], points_sub[:, 2], c=colors_sub, s=2, alpha=0.8)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("NeuralMP Point Cloud")

plt.show()
