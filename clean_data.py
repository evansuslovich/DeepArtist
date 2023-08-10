import pandas as pd
# Assuming you have already loaded the DataFrame from the CSV file
import os

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

folder = get_files_in_folder("Data")
for csv_file in folder:
    try:
        df = pd.read_csv("Data/" + csv_file)

        # Create a condition to filter the rows based on the given conditions
        condition = (
                (df['Red'] >= 15) & (df['Green'] >= 15) & (df['Blue'] >= 15) &
                (df['Red'] <= 240) & (df['Green'] <= 240) & (df['Blue'] <= 240) &
                (abs(df['Red'] - df['Green']) > 15) &
                (abs(df['Red'] - df['Blue']) > 15) &
                (abs(df['Green'] - df['Blue']) > 15)
        )
        # Apply the condition to the DataFrame
        df_filtered = df[condition]

        # Save the filtered DataFrame as a new CSV file
        df_filtered.to_csv("New_Data/New_" + csv_file, index=False)
    except UnicodeDecodeError as e:
        print(e)

