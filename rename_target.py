import pandas as pd
import os


# def get_files_in_folder(folder_path):
#     try:
#         # Get a list of files and directories in the specified folder
#         files = os.listdir(folder_path)
#         # Filter out directories, keep only files
#         files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]
#
#         return files
#     except OSError as e:
#         print(f"Error accessing folder: {e}")
#         return []
#
# csv_files = list(map(lambda item: 'Test_Quantify/' + item, get_files_in_folder('Test_Quantify')))
#
# df_csv_concat = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=False)
#
# print(df_csv_concat)
#
# dictionary = {
#     'Academic Art': 1,
#     'Art Nouveau': 2,
#     'Baroque': 3,
#     'Expressionism': 4,
#     'Japanese Art': 5,
#     'Neoclassicism': 6,
#     'Primitivism': 7,
#     'Realism': 8,
#     'Renaissance': 9,
#     'Rococo': 10,
#     'Romanticism': 11,
#     'Symbolism': 12,
#     'Western Medieval': 13,
# }
#
# df_csv_concat['Genre'] = df_csv_concat['Genre'].map(dictionary)
# df_csv_concat['Genre'] = df_csv_concat['Genre'].replace(dictionary)


# art_test = pd.read_csv('art_test.csv')
# art_train = pd.read_csv('art_train.csv')
#
#
# art_test = art_test.drop(columns=art_test.columns[0])
#art_test.to_csv("art_test.csv", index=False)

#df_csv_concat.to_csv('art_test.csv')
