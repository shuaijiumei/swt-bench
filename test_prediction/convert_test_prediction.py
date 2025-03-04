import os
import json

def convert_jsonl_to_json(folder_path):
    save_dir = '/mnt/d/vscodeProject/swt-bench/test_prediction/agentless_test'
    all_items = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.jsonl'):
                jsonl_file_path = os.path.join(root, filename)
                folder_name = os.path.basename(root)  # Get the name of the parent folder
                with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
                    for line in jsonl_file:
                        item = json.loads(line)
                        all_items.append((item, folder_name))  # Store item with its folder name

    # Save each item as a separate JSON file in the corresponding folder
    for index, (item, folder_name) in enumerate(all_items):
        json_file_name = f'test_patch_{index%80}.json'
        json_file_path = os.path.join(save_dir, folder_name, json_file_name)  # Save in the folder name
        os.makedirs(os.path.join(save_dir, folder_name), exist_ok=True)  # Create folder if it doesn't exist
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump([item], json_file, ensure_ascii=False, indent=4)

# Example usage
convert_jsonl_to_json('/mnt/d/vscodeProject/Agentless/results/swe-bench-lite/reproduction_test_samples/reproduce_test_individual/')
