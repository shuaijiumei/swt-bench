import os

def categorize_folders(base_path):
    categorized_folders = {}
    
    for root, dirs, _ in os.walk(base_path):
        for dir_name in dirs:
            group_name = dir_name.split('__')[0] # Replace underscores with double underscores
            if group_name not in categorized_folders:
                categorized_folders[group_name] = []
            categorized_folders[group_name].append(dir_name)

    # Move categorized folders to new directories
    for group_name, folder_names in categorized_folders.items():
        new_group_path = os.path.join(base_path, group_name)
        os.makedirs(new_group_path, exist_ok=True)  # Create the new group directory if it doesn't exist
        for folder_name in folder_names:
            old_folder_path = os.path.join(base_path, folder_name)
            new_folder_path = os.path.join(new_group_path, folder_name)
            os.rename(old_folder_path, new_folder_path)  # Move the folder to the new group directory

    return categorized_folders

# Example usage
folder_path = '/mnt/d/vscodeProject/swt-bench/test_prediction/agentless_test'
categorized = categorize_folders(folder_path)
print(categorized)
