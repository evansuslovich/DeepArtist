import os
import csv
import numpy as np
from PIL import Image


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


def most_common_rgb(image_path):
    # Open Paddington and make sure he is RGB - not palette
    im = Image.open(image_path).convert('RGB')

    # Make into Numpy array
    na = np.array(im)

    # Flatten the numpy array to count individual RGB combinations
    flattened_na = na.reshape(-1, 3)

    # Get unique RGB values and their counts
    unique_vals, counts = np.unique(flattened_na, axis=0, return_counts=True)

    # Find the index of the most common RGB value
    most_common_index = np.argmax(counts)

    # Get the most common RGB value
    return unique_vals[most_common_index]


if __name__ == "__main__":
    archive = get_folders_in_folder('Archive')
    archive.remove("Expressionism")
    for folder in archive:
        print(folder)
        folder_path = 'Archive/' + folder + "/" + folder + "/"
        with open(folder + "_Data.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Filename', 'Red', 'Green', 'Blue'])

        files_in_folder = get_files_in_folder(folder_path)
        for file_index in range(800):
            color = list(most_common_rgb(folder_path + files_in_folder[file_index]))
            print(files_in_folder[file_index], color[0], color[1], color[2])
            with open(folder + "_Data.csv", 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([files_in_folder[file_index], round(color[0]), round(color[1]), round(color[2])])
