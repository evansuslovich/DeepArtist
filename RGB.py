import csv
import os
from PIL import Image
from collections import Counter


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


def get_most_common_rgb(image_path, num_colors=1):
    try:
        img = Image.open(image_path)

        # Check if the image mode is RGB; if not, convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        width, height = img.size
        rgb_values = []
        for y in range(height):
            for x in range(width):
                try:
                    r, g, b = img.getpixel((x, y))
                except ValueError:
                    print("oh nah")
                else:
                    rgb_values.append((r, g, b))

        # Count the occurrences of each RGB value
        color_counter = Counter(rgb_values)

        # Get the most common RGB values with their counts
        most_common_colors = color_counter.most_common(num_colors)

        return most_common_colors
    except IOError as e:
        print(f"Error: Unable to open or read the image file '{image_path}'.")
        return None


if __name__ == "__main__":
    folder_path = 'Expressionism'

    with open('output.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'Red', 'Green', 'Blue'])

    files_in_folder = get_files_in_folder(folder_path)

    for file in files_in_folder:
        most_common_colors = get_most_common_rgb(os.path.join(folder_path, file))
        for color, count in most_common_colors:
            print(file, color, count)
            with open('output.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([file, round(color[0]), round(color[1]), round(color[2])])
