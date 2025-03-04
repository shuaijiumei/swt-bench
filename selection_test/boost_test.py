import os

from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import numpy as np
import json
import re
import random
from datasets import load_dataset
from tqdm import tqdm

def cluster_errors_with_dbscan(errors_dict, eps=0.5, min_samples=5):
  """
  使用 DBSCAN 对测试代码的报错信息进行聚类
  
  参数：
  - errors_dict: dict, 需要聚类的报错信息字典，键为错误来源，值为错误信息。
  - eps: float, DBSCAN 的距离阈值，默认是 0.5。
  - min_samples: int, DBSCAN 中定义核心点的最小样本数，默认是 5。
  
  返回：
  - clusters: dict, 每个聚类的错误信息，键为聚类标签，值为对应的错误来源列表。
  """
  # Step 1: 加载 Sentence-BERT 模型
  model = SentenceTransformer('all-MiniLM-L6-v2')
  
  # Step 2: 获取每个报错信息的向量表示
  errors = list(errors_dict.values())
  embeddings = model.encode(errors)
  
  # Step 3: 使用 DBSCAN 聚类
  db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
  labels = db.fit_predict(embeddings)
  
  # Step 4: 根据聚类标签将错误信息分组
  clusters = {}
  for idx, label in enumerate(labels):
    if label not in clusters:
      clusters[label] = []
    clusters[label].append(list(errors_dict.keys())[idx])
  
  return clusters

def normalize_string(text: str) -> str:
    """
    处理字符串中的不可见字符，返回规范化后的字符串
    
    处理内容包括：
    1. 去除首尾空白字符
    2. 统一换行符
    3. 移除零宽字符
    4. Unicode 规范化
    5. 替换连续的空白字符为单个空格
    """
    import re
    import unicodedata
    
    if not isinstance(text, str):
        text = str(text)
    
    # # 4. 替换连续的空白字符为单个空格
    text = re.sub(r'\s+', ' ', text)
    
    # # 5. 去除首尾空白
    # text = text.strip()
    
    return text


def get_test_output_content(instance_id):
  base_path = '/mnt/d/vscodeProject/swt-bench/run_instance_swt_logs/only_evaluation/agentless_test_boost'
  repo_name = instance_id.split('__')[0]
  test_output_dict = {}
  base_path = os.path.join(base_path, repo_name)
  for root, dirs, files in os.walk(base_path):
    if os.path.basename(root) == instance_id:
      test_output_path = os.path.join(root, 'test_output.txt')
      exec_spec_path = os.path.join(root, 'exec_spec.json')
      
      if os.path.exists(exec_spec_path):
        with open(exec_spec_path, 'r', encoding='utf-8') as f:
          exec_spec_dict = json.load(f)
          test_command = exec_spec_dict['test_command']
      
      if os.path.exists(test_output_path):
        with open(test_output_path, 'r', encoding='utf-8') as f:
          content = f.read()
          # 按行分割内容
          lines = content.splitlines()
          
          # 找到包含 test_command 的行号
          test_command_line = -1
          coverage_line = -1
          for i, line in enumerate(lines):
            if normalize_string(test_command) in normalize_string(line):
              test_command_line = i
            if "+ cat coverage.cover" in line:
              coverage_line = i
              break
          
          if test_command_line == -1:
            print(f"Test command not found in {test_output_path}")
          elif coverage_line != -1:
            # 提取测试命令行到覆盖率行之间的内容
            test_output = "\n".join(lines[test_command_line+2:coverage_line])
            test_output_dict[test_output_path] = test_output
  
  return test_output_dict

def write_test_output_to_tmp_file(instance_id, index, content, path, cleaned_dict):
    # 创建临时文件夹
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    
    # 写入文件
    output_file = os.path.join('tmp', f'{instance_id}_{index}.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    cleaned_dict[path] = content

def strip_ansi(text: str) -> str:
    """
    去除字符串中的 ANSI 转义序列
    """
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def clean_test_output_dict(test_output_dict, instance_id):
  group_1 = ['mwaskom', 'astropy', 'pallets', 'psf','pylint-dev', 'scikit-learn', 'pydata', 'pytest-dev', 'matplotlib', 'sphinx-doc']
  group_2 = ['django']
  cleaned_dict = {}
  repo_name = instance_id.split('__')[0]

  for index, (path, content) in enumerate(test_output_dict.items()):
    if 'AssertionError' not in content:
      continue
    # 根据 instance_id 区分不同的清理策略
    if repo_name in group_1:
        # 1. 先找到 short test summary info 后的内容
        short_test_summary_index = content.find('short test summary info')
        if short_test_summary_index == -1:
          print(f"Short test summary info not found in {path}")
          continue
        short_info_content = content[short_test_summary_index:]
        if repo_name == 'pydata' or 'FAILED' in short_info_content:
          # 提取以 E 开头的行并去掉 E
          error_lines = []
          for line in content.splitlines():
              if strip_ansi(line).startswith('E '):
                  error_lines.append(strip_ansi(line)[2:])  # 去掉 'E ' 前缀
          content = '\n'.join(error_lines)
          cleaned_dict[path] = content

          # Debug 用
          # write_test_output_to_tmp_file(instance_id, index, content, path, cleaned_dict)
        
        
    elif repo_name in group_2:
        # django 项目的测试输出通常包含 'FAILED' 和具体的失败信息
        if 'FAILED' not in content:
            continue
        match = re.search(r"Ran \d+ tests? in \d+\.\d+s", content)
        if match:
            test_output_content = content[:match.start()]
            cleaned_dict[path] = test_output_content
  
  return cleaned_dict


# 输出聚类结果
def print_cluster_results(clusters, cleaned_result):
  # 按照每个聚类中错误数量排序
  sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
  
  for label, cluster_errors in sorted_clusters:
    print(f"Cluster {label} (包含 {len(cluster_errors)} 个错误):")
    cluster_content = []
    for index,error in enumerate(cluster_errors):
      print(f"测试函数 {index}:")
      print(f"  {error}")
      cluster_content.append(cleaned_result[error])
    print("\n".join(cluster_content))
    print()

def process_clusters_and_save_results(cleaned_result, instance_id, prediction_base_path='/mnt/d/vscodeProject/swt-bench/test_prediction/agentless_test_merge'):
    """
    处理聚类结果并保存到results.json文件
    
    参数:
    - cleaned_result: dict, 清理后的测试输出结果
    - instance_id: str, 实例ID
    - prediction_base_path: str, 预测结果的基础路径
    """
    clusters = {}
    if cleaned_result:
        clusters = cluster_errors_with_dbscan(cleaned_result, eps=0.3, min_samples=2)
        if clusters:
            # 获取最大的聚类
            max_cluster = max(clusters.items(), key=lambda x: len(x[1]))
            # 随机选择一个错误
            random_error = random.choice(max_cluster[1])

            # 从 random_error 中提取出错误ID
            error_path = random_error.split(':')[0].strip()
            error_id = error_path.split('/')[-4]
            instance_path = os.path.join(prediction_base_path, instance_id.split('__')[0], f'merged_test_patch_{error_id}.json')
            
            with open(instance_path, 'r', encoding='utf-8') as f:
                instance_content = json.load(f)

            # 添加新的结果
            for item in instance_content:
                if item['instance_id'] == instance_id:
                    return item
            
    return None

# Example usage
if __name__ == "__main__":

# Login using e.g. `huggingface-cli login` to access this dataset
  ds = load_dataset("nmuendler/SWT-bench_Lite_bm25_27k_zsp", split='test')
  # test_project_list = ['mwaskom', 'astropy', 'pallets', 'psf','pylint-dev', 'scikit-learn']
  test_project_list = ['django', 'pydata', 'sphinx-doc', 'pytest-dev', 'matplotlib']
  instance_ids = [item['instance_id'] for item in ds if item['instance_id'].split('__')[0] in test_project_list]
  
  result_json_path = 'results_other.json'
  failed_instances_path = 'failed_instances_other.json'
  not_found_instances_path = 'not_found_instances_other.json'
  
  result_json = []
  failed_instances = []
  not_found_instances = []

  try:
    
    for instance_id in tqdm(instance_ids, desc="处理实例", unit="个"):
      # 检查是否为 sympy 项目
      if instance_id.split("__")[0] == "sympy":
        # 对于 sympy 项目直接从 json 文件中抽奖
        project_name = instance_id.split("__")[0]
        # 随机生成一个错误ID
        error_id = random.randint(0, 79)
        instance_path = os.path.join('/mnt/d/vscodeProject/swt-bench/test_prediction/agentless_test_merge', 
                                   project_name, f'merged_test_patch_{error_id}.json')
        
        try:
          with open(instance_path, 'r', encoding='utf-8') as f:
            instance_content = json.load(f)
            
          # 添加新的结果
          for item in instance_content:
            if item['instance_id'] == instance_id:
              result_json.append(item)
              continue
              
        except FileNotFoundError:
          failed_instances.append(instance_id)
          print(f"需要重新生成测试用例: {instance_id}")
          continue
          
      else:
        # 非 sympy 项目按原流程处理
        result = get_test_output_content(instance_id)
        if not result:
          not_found_instances.append(instance_id)
          print(f"测试输出不存在: {instance_id}")
          continue

        cleaned_result = clean_test_output_dict(result, instance_id)
        item = process_clusters_and_save_results(cleaned_result, instance_id)
        if item:
          result_json.append(item)
        else:
          failed_instances.append(instance_id)
          print(f"需要重新生成测试用例: {instance_id}")
      

    with open(result_json_path, 'w', encoding='utf-8') as f:
      json.dump(result_json, f, ensure_ascii=False, indent=2)
    with open(failed_instances_path, 'w', encoding='utf-8') as f:
      json.dump(failed_instances, f, ensure_ascii=False, indent=2)
    with open(not_found_instances_path, 'w', encoding='utf-8') as f:
      json.dump(not_found_instances, f, ensure_ascii=False, indent=2)
    
  except FileNotFoundError as e:
    print(e)