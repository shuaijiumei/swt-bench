#!/bin/bash

INSTANCE_ID=django__django-10914

patch_dirs=$(ls -d test_prediction/${INSTANCE_ID}/test_patch_* | wc -l)

for i in $(seq 0 $(($patch_dirs - 1))); do
  json_files=$(ls test_prediction/${INSTANCE_ID}/test_patch_${i}/test_patch_*.json | wc -l)
  for j in $(seq 0 $(($json_files - 1))); do
    python covert.py ./test_prediction/${INSTANCE_ID}/test_patch_${i}/test_patch_${j}.json

    python -m src.main \
      --dataset_name princeton-nlp/SWE-bench_Lite \
      --predictions_path test_prediction/${INSTANCE_ID}/test_patch_${i}/test_patch_${j}.json \
      --max_workers 2 \
      --run_id evaluation/${INSTANCE_ID}/test_patch_${i}/test_patch_${j} \
      --only_run_test True
  done
done

