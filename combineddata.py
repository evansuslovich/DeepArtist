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

#print(get_files_in_folder('Test_Quantify'))

filtered_list = [s for s in get_files_in_folder('Test_Quantify') if "GlCM" in s]
filtered_list2 = [s for s in get_files_in_folder('Test_Quantify') if "GlCM" not in s]
#print(filtered_list)
#print(filtered_list2)

#
csv_files = list(map(lambda item: 'Test_Quantify/' + item, filtered_list2))
glcm = list(map(lambda item: 'Test_Quantify/' + item, filtered_list))
#print(csv_files)
# #
# #
df_csv_concat = pd.concat([pd.read_csv(file) for file in csv_files ], ignore_index=True)
glcm_concat = pd.concat([pd.read_csv(file) for file in glcm ], ignore_index=True)
#print(df_csv_concat)
#
test_data = pd.merge(df_csv_concat, glcm_concat, on='Filename', how='inner')
#print(test_data)

dictionary = {
    'Academic Art': 1,
    'Art Nouveau': 2,
    'Baroque': 3,
    'Expressionism': 4,
    'Japanese Art': 5,
    'Neoclassicism': 6,
    'Primitivism': 7,
    'Realism': 8,
    'Renaissance': 9,
    'Rococo': 10,
    'Romanticism': 11,
    'Symbolism': 12,
    'Western_Medieval': 13,
}

test_data['Genre'] = test_data['Genre'].map(dictionary)
test_data['Genre'] = test_data['Genre'].replace(dictionary)


#art_test = test_data.drop(columns=test_data.columns[0])

print(test_data)
#
#
#
test_data.to_csv('art_test.csv')
#
# #print(df_csv_concat['Genre'].value_counts())
