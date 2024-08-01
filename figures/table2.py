"""
Method vs Applicability, FtX, FtP and PtP
"""
import functools

import fire
from pathlib import Path
import json

def fold_over_reports(start, method, model_name, run_id, instance_log_path: Path):
    run_path = instance_log_path / run_id / model_name
    fold = start
    for dir in run_path.iterdir():
        if not dir.is_dir():
            continue
        report = dir / "report.json"
        if not report.exists():
            continue
        with open(report) as f:
            report_data = json.load(f)
        instance_id = dir.name
        fold = method(fold, report_data[instance_id])
    return fold


def applied_count(prev, report):
    prev += report["patch_successfully_applied"]
    return prev

def ftp_count(prev, report):
    prev += report["resolved"]
    return prev

report_applied_count = functools.partial(fold_over_reports, 0, applied_count)
report_ftp_count = functools.partial(fold_over_reports, 0, ftp_count)


def main(instance_log_path: str = "./run_instance_swt_logs", total_instance_count: int = 300):
    instance_log_path = Path(instance_log_path)
    if not instance_log_path.exists():
        raise FileNotFoundError(f"Instance log directory not found at {instance_log_path}")
    methods = [
        ("gpt-4-1106-preview", "zsb__gpt-4-1106-preview__bm25_27k_cl100k__seed=0,temperature=0", r"\zsb"),
        ("gpt-4-1106-preview", "zsp__gpt-4-1106-preview__bm25_27k_cl100k__seed=0,temperature=0", r"\zsp"),
        # TODO libro,
        ("gpt-4-1106-preview", "acr__gpt-4-1106-preview", r"\acr"),
        ("aider--gpt-4-1106-preview", "aider_gpt-4-1106-preview", r"\aider"),
        ("gpt4__SWE-bench_Lite__default_test_demo3__t-0.00__p-0.95__c-3.00__install-1", "swea__gpt-4-1106-preview", r"\swea"),
        ("gpt4__SWE-bench_Lite__default_test_demo4__t-0.00__p-0.95__c-3.00__install-1", "sweap__gpt-4-1106-preview", r"\sweap"),
    ]

    print("Method & Applicability & FtX & FtP & PtP")
    for model, run_id, name, in methods:
        applied = report_applied_count(model, run_id, instance_log_path)
        ftp = report_ftp_count(model, run_id, instance_log_path)
        print(f"{name} & {100*applied/total_instance_count:.1f} & {100*ftp/total_instance_count:.1f} & {0:.1f} & {0:.1f}")

if __name__ == "__main__":
    fire.Fire(main)