import os
import json
from collections import defaultdict
base_dir_list = [name for name in os.listdir('./agentless_test') if os.path.isdir(os.path.join('./agentless_test', name))]

def merge_json_files(base_dir='./agentless_test/django', output_dir='./agentless_test/django_solo'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_files_dict = defaultdict(list)

    # Traverse through all directories and files
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json') and file.startswith('test_patch_'):
                num = file.split('_')[-1].split('.')[0]  # Extract the number from the filename
                json_files_dict[num].append(os.path.join(root, file))

    # Merge files with the same num
    for num, file_paths in json_files_dict.items():
        merged_data = []
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                data = json.load(f)
                merged_data.extend(data)  # Assuming the JSON files contain lists

        output_file_path = os.path.join(output_dir, f'merged_test_patch_{num}.json')
        with open(output_file_path, 'w') as f:
            json.dump(merged_data, f, indent=4)


for base_dir in base_dir_list:
    merge_json_files(f'./agentless_test/{base_dir}', f'./agentless_test_merge/{base_dir}')
