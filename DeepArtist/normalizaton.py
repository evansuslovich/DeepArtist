import os

def get_folders_in_folder(folder_path):
    try:
        folders = os.listdir(folder_path)
        # folders.remove('.DS_Store')
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



if __name__ == '__main__':
    folders = get_folders_in_folder("./Data")
    for folder in folders: 
        print(folder)
        print(len(get_files_in_folder("./Data/" + folder)))
        print()