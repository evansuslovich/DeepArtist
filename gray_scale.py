import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color
from PIL import Image

# Load the grayscale image
image = Image.open('archive/Academic_Art/Academic_Art/232331.jpg').convert('L')  # Convert to grayscale
image_array = np.array(image)

# Calculate GLCM
distance = 50
angles = np.array([0, 45, 90, 135, 180, 225, 270, 315])
angles_radians = np.radians(angles)  # Convert angles to radians
glcm = graycomatrix(image_array, [distance], angles_radians, symmetric=True, normed=True)

# Calculate contrast from GLCM
contrast = graycoprops(glcm, prop='contrast')

print("Contrast:", contrast)
