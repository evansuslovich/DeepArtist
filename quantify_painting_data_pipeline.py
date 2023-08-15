import os
import csv
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from skimage.feature import graycomatrix, graycoprops



def get_folders_in_folder(folder_path):
    try:
        folders = os.listdir(folder_path)
        folders.remove('.DS_Store')
        return folders
    except OSError as e:
        print(f"Error accessing folder: {e}")
        return []

def get_files_in_folder(folder_path):
    try:
        # Get a list of files and directories in the specified folder
        files = os.listdir(folder_path)
        # Filter out directories, keep only files
        files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]

        return files
    except OSError as e:
        print(f"Error accessing folder: {e}")
        return []


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

        # Handle small standard deviations to avoid division by near-zero values
        epsilon = 1e-6  # Small constant to prevent division by zero
        adjusted_std_dev = max(std_dev, epsilon)

        skewness = np.mean(((channel - mean) / adjusted_std_dev) ** 3)
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

def generate_glcm(image):
    # Calculate GLCM
    distance = 50
    angles = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    angles_radians = np.radians(angles)  # Convert angles to radians
    glcm = graycomatrix(image_array, [distance], angles_radians, symmetric=True, normed=True)

    # Calculate contrast from GLCM
    contrast = graycoprops(glcm, prop='contrast')

    return contrast[0]

if __name__ == "__main__":
    archive = get_folders_in_folder('archive')
    archive.remove("Expressionism")
    # Evan: Academic_Art, Art_Nouveau, Baroque, Expressionism, Japanese_Art, Neoclassicism, Primitivism,
    # Sofie: Realism, Renaissance, Rococo, Symbolism, Western_Medieval, 
    for folder in archive:
        print(folder)
        folder_path = 'archive/' + folder + "/" + folder + "/"
        with open("Test_Quantify/" + folder + "_GlCM_Data.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['Filename',
                 '0 Degrees',
                 '45 Degrees',
                 '90 Degrees',
                 '135 Degrees',
                 '180 Degrees',
                 '225 Degrees',
                 '270 Degrees',
                 '315 Degrees',
                 # 'Channel 1 Mean',
                 # 'Channel 1 Standard Deviation',
                 # 'Channel 1 Skewness',
                 # 'Channel 2 Mean',
                 # 'Channel 2 Standard Deviation',
                 # 'Channel 2 Skewness',
                 # 'Channel 3 Mean',
                 # 'Channel 3 Standard Deviation',
                 # 'Channel 3 Skewness',
                 # 'Color Richness',
                 # 'Color Palette 1 Red',
                 # 'Color Palette 1 Green',
                 # 'Color Palette 1 Blue',
                 # 'Color Palette 2 Red',
                 # 'Color Palette 2 Green',
                 # 'Color Palette 2 Blue',
                 # 'Color Palette 3 Red',
                 # 'Color Palette 3 Green',
                 # 'Color Palette 3 Blue',
                 # 'Color Palette 4 Red',
                 # 'Color Palette 4 Green',
                 # 'Color Palette 4 Blue',
                 # 'Color Palette 5 Red',
                 # 'Color Palette 5 Green',
                 # 'Color Palette 5 Blue',
                 ])

        files_in_folder = get_files_in_folder(folder_path)
        for file_index in range(801,1001):
            print(files_in_folder[file_index])
            image_path = folder_path + files_in_folder[file_index]
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            image_array = np.array(image)
            glcm = generate_glcm(image_array)
            # image = Image.open(image_path).convert('RGB')
            # converted_image = np.array(image).reshape(-1, 3)
            # color_moments = calculate_color_moments(image)
            # color_richness = calculate_color_richness(converted_image, image)
            # color_palette = generate_color_palette(converted_image)
            with open("Test_Quantify/" + folder + "_GlCM_Data.csv", 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        files_in_folder[file_index],
                        glcm[0],
                        glcm[1],
                        glcm[2],
                        glcm[3],
                        glcm[4],
                        glcm[5],
                        glcm[6],
                        glcm[7]
                        # color_moments[0], color_moments[1], color_moments[2],
                        # color_moments[3], color_moments[4], color_moments[5],
                        # color_moments[6], color_moments[7], color_moments[8],
                        # color_richness,
                        # color_palette[0][0], color_palette[0][1], color_palette[0][2],
                        # color_palette[1][0], color_palette[1][1], color_palette[1][2],
                        # color_palette[2][0], color_palette[2][1], color_palette[2][2],
                        # color_palette[3][0], color_palette[3][1], color_palette[3][2],
                        # color_palette[4][0], color_palette[4][1], color_palette[4][2],
                    ]
                )
