import pandas as pd

# Assuming you have already loaded the DataFrame from the CSV file
path = "path"
df = pd.read_csv(path)

# Create a condition to filter the rows based on the given conditions
condition = (
    (df['Red'] >= 20) & (df['Green'] >= 20) & (df['Blue'] >= 20) &
    (df['Red'] <= 240) & (df['Green'] <= 240) & (df['Blue'] <= 240)
)

# Apply the condition to the DataFrame
df_filtered = df[condition]

# Save the filtered DataFrame as a new CSV file
df_filtered.to_csv("New_" + path, index=False)
