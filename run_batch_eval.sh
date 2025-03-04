#!/bin/bash

# 从文件读取 instance_ids，每行一个
input_file="/mnt/d/vscodeProject/swt-bench/instance_ids.txt "

while IFS= read -r id
do
    # 跳过空行和注释行
    [[ -z "$id" || "$id" =~ ^#.*$ ]] && continue
    
    echo "Running evaluation for instance: $id"
    python -m src.main \
        --predictions_path gold \
        --max_workers 1 \
        --instance_ids "$id" \
        --run_id validate-gold/$id
    
    sleep 1
done < "$input_file" 