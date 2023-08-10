# After processing all images, read the CSV data and create a 3D graph
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('output.csv')

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract data from the DataFrame
red_values = df['Red']
green_values = df['Green']
blue_values = df['Blue']

# Plot the RGB values in the 3D graph
ax.scatter(red_values, green_values, blue_values)

# Add labels to the axes
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')

# Add titles for each point with the corresponding filename

plt.show()