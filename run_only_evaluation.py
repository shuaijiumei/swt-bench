import os
import subprocess
import glob
from tqdm import tqdm  # Ensure tqdm is imported for progress bar
import time
import logging
import shutil
from prettytable import PrettyTable

# Configure logging to append to the log file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create a file handler that appends to the log file
file_handler = logging.FileHandler('./logs/evaluation.log', mode='a')  # Append mode
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Get the root logger and add the file handler
logger = logging.getLogger()
logger.addHandler(file_handler)

def collect_instance_ids(base_dir='test_prediction/agentless_test'):
    instance_ids = []
    for root, dirs, files in os.walk(base_dir):
        if files:  # If there are files in the directory
            instance_id = os.path.basename(root)
            if instance_id not in instance_ids:
                instance_ids.append(instance_id)

    from collections import defaultdict

    grouped_instance_ids = defaultdict(list)
    for instance_id in instance_ids:
        group_key = instance_id.split("__")[0]
        grouped_instance_ids[group_key].append(instance_id)

    
    return grouped_instance_ids

# Function to run the evaluation
def run_evaluation_instance(json_file, project_name):
    subprocess_logger = logging.getLogger('subprocess_logger')
    subprocess_handler = logging.FileHandler(f'./logs/subprocess_evaluation.log', mode='a')
    subprocess_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    subprocess_logger.addHandler(subprocess_handler)
    subprocess_logger.setLevel(logging.INFO)

    try:
        # 动态调整 max_workers，如果第一次运行，就设置为 2，否则设置为 12
        json_num = os.path.basename(json_file).replace(".json", "").split("_")[-1]
        max_workers = '2' if json_num == '0' else '4'
        
        result = subprocess.run([
            'python', '-m', 'src.main',
            '--dataset_name', 'princeton-nlp/SWE-bench_Lite',
            '--predictions_path', json_file,
            '--max_workers', max_workers,
            '--run_id', f'only_evaluation/agentless_test_boost/{project_name}/{json_num}',
            '--only_run_test', 'True'
        ], capture_output=True, text=True)

        subprocess_logger.info(f"Output for {json_file}:{result.stdout}")
        if result.stderr:
            subprocess_logger.error(f"Error for {json_file}:{result.stderr}")
    except Exception as e:
        subprocess_logger.error(f"Exception occurred while processing {json_file}: {e}")
    finally:
        subprocess_logger.removeHandler(subprocess_handler)
        subprocess_handler.close()  # Close the handler to prevent leaks

def should_run_evaluation(json_file, project_name, group_counts):
    """
    该函数用于判断是否需要运行评估。
    swt-bench 中开启了 cache log dir，如果有 test_output.txt 文件，则不运行评估。
    所以在这里，我只需要判断整个 json 文件是否需要运行评估。
    """
    
    # Construct the test path based on the project name and JSON file name
    test_path = f'run_instance_swt_logs/only_evaluation/agentless_test_boost/{project_name}/{os.path.basename(json_file).replace(".json", "").split("_")[-1]}'
    
    # Check if the test path exists; if not, log the information and indicate that evaluation should run
    if not os.path.exists(test_path):
        print(f"Running evaluation for {json_file} as test_path does not exist.")
        return True

    # Define the path where the test outputs are expected to be found
    test_output_path = os.path.join(test_path, 'pred_pre__agentless_test')
    
    # List all directories in the test output path
    folders = [f for f in os.listdir(test_output_path) if os.path.isdir(os.path.join(test_output_path, f))]
    
    # Count the number of folders found
    folder_count = len(folders)
    
    # Check if the number of folders matches the expected count for the project
    folder_number_flag = folder_count == group_counts[project_name]
    
    # Check if all folders contain the 'test_output.txt' file
    all_have_test_output_flag = all(os.path.exists(os.path.join(test_output_path, folder, 'test_output.txt')) for folder in folders)

    # If both conditions are met, log the information and skip the evaluation
    if folder_number_flag and all_have_test_output_flag:
        print(f"Skipping folder: {project_name}_{os.path.basename(json_file).replace('.json', '').split('_')[-1]}.json because it has the correct number of folders and all have test_output.txt")
        return False

    return True


def run_evaluation(project_name, group_counts):
    json_files = glob.glob(f'/mnt/d/vscodeProject/swt-bench/test_prediction/agentless_test_merge/{project_name}/merged_test_patch_*.json')
    # Use tqdm with total parameter to show the correct progress
    for json_file in tqdm(json_files, desc="Processing JSON files", unit="file", total=len(json_files)):
        try:
            if should_run_evaluation(json_file, project_name, group_counts):
                run_evaluation_instance(json_file, project_name)  # Start the subprocess
                time.sleep(10)
        except Exception as e:
            logger.error(f"Error processing file {json_file}: {e}")


def main():
    INSTANCE_IDS_LIST = collect_instance_ids()
    group_counts = {group_key: len(instance_ids) for group_key, instance_ids in INSTANCE_IDS_LIST.items()}

    table = PrettyTable()
    table.field_names = ["Group Key", "Count"]
    for group_key, count in group_counts.items():
        table.add_row([group_key, count])
    print(table)
    # logger.info("\n" + str(table))

    skip_count = 0  # Number of projects to skip
    with tqdm(total=len(INSTANCE_IDS_LIST) - skip_count, desc="Running evaluations for projects", unit="project") as pbar:
        for i, project_name in enumerate(list(INSTANCE_IDS_LIST.keys())):
            if i < skip_count:
                continue  # Skip the first few projects
            run_evaluation(project_name, group_counts)
            pbar.update(1)

if __name__ == "__main__":
    main()