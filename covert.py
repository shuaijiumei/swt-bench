import json
import sys

def convert_json(file_path):
  with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
  
  if 'test_patch' in data:
    data['model_patch'] = data.pop('test_patch')
    keys_to_keep = ['model_patch', 'instance_id', 'model_name_or_path']
    data = [{key: data[key] for key in keys_to_keep if key in data}]
  
  with open(file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python covert.py <json_file_path>")
    sys.exit(1)
  
  json_file_path = sys.argv[1]
  convert_json(json_file_path)