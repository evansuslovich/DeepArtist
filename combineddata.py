import pandas as pd
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

print(get_files_in_folder('Test_Quantify'))


csv_files = list(map(lambda item: 'Test_Quantify/' + item, get_files_in_folder('Test_Quantify')))
print(csv_files)


df_csv_concat = pd.concat([pd.read_csv(file) for file in csv_files ], ignore_index=True)


#df_csv_concat.to_csv('art_test.csv')

#print(df_csv_concat['Genre'].value_counts())
