from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle


def calculate_color_moments(image):
    # Convert the image to the Lab color space
    lab_image = image.convert('LAB')
    lab_array = np.array(lab_image)

    # Split the Lab image into its channels
    L, a, b = lab_array[:, :, 0], lab_array[:, :, 1], lab_array[:, :, 2]

    # Calculate mean, standard deviation, and skewness for each channel
    moments = []
    for channel in [L, a, b]:
        mean = np.mean(channel)
        std_dev = np.std(channel)
        skewness = np.mean(((channel - mean) / std_dev) ** 3)
        moments.extend([mean, std_dev, skewness])

    return moments


def generate_color_palette(image, n_colors=5, n_init=10, sample_size=1000):
    # Load and preprocess image

    # Shuffle the pixels to avoid any bias
    image_sample = shuffle(image, random_state=0)[:sample_size]
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_colors, n_init=n_init, random_state=0).fit(image_sample)
    # Get the cluster centers (colors)
    cluster_centers = kmeans.cluster_centers_
    # Sort cluster centers by frequency
    sorted_clusters = cluster_centers[np.argsort(np.bincount(kmeans.labels_, minlength=n_colors))]

    return sorted_clusters


# how we are calculating the color richness is based off the unique colors
def calculate_color_richness(colors, image):
    unique_colors = np.unique(colors, axis=0)
    richness = len(unique_colors) / float(image.size[0] * image.size[1])
    return richness

# Load an image using Pillow
image_path = 'archive/Baroque/Baroque/186638.jpg'
converted_image = np.array(Image.open(image_path)).reshape(-1, 3)
image = Image.open(image_path)
color_moments = calculate_color_moments(image)
print("Color Moments:", color_moments)
color_richness = calculate_color_richness(converted_image, image)
print("Color Richness:", color_richness)
color_palette = generate_color_palette(converted_image)
print(color_palette)
