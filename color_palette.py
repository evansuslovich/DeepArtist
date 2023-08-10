import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle


# Load and preprocess image
image = plt.imread('archive/Baroque/Baroque/186638.jpg')
original_shape = image.shape
image = image.reshape(-1, 3)  # Reshape to (n_pixels, 3) for RGB values

# Shuffle the pixels to avoid any bias
image_sample = shuffle(image, random_state=0)[:1000]

# Perform K-means clustering
n_colors = 5  # Number of colors in the palette
n_init = 10   # Explicitly set the number of initializations
kmeans = KMeans(n_clusters=n_colors, n_init=n_init, random_state=0).fit(image_sample)

# Get the cluster centers (colors)
cluster_centers = kmeans.cluster_centers_

# Sort cluster centers by frequency
sorted_clusters = cluster_centers[np.argsort(np.bincount(kmeans.labels_, minlength=n_colors))]

# Create a combined plot
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Display the original image
axes[0].imshow(image.reshape(original_shape))
axes[0].set_title("Original Image")
axes[0].axis('off')

# Display the color palette image
# Display the color palette image
palette_image = sorted_clusters.reshape((1, n_colors, 3)).astype(np.uint8)
axes[1].imshow(palette_image)
axes[1].set_title("Color Palette")
axes[1].axis('off')

print(sorted_clusters)

plt.tight_layout()
plt.show()
