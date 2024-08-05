"""
Method vs Applicability, FtX, FtP and PtP
main
"""

from tabulate import tabulate
import fire
from pathlib import Path

from figures.util import *


def main(instance_log_path: str = "./run_instance_swt_logs", total_instance_count: int = 279, format="github", max_turns=20):
    instance_log_path = Path(instance_log_path)
    if not instance_log_path.exists():
        raise FileNotFoundError(f"Instance log directory not found at {instance_log_path}")
    methods = [
        ("gpt4__SWE-bench_Lite__default_test_demo3__t-0.00__p-0.95__c-3.00__install-1", "swea__gpt-4-1106-preview", r"\swea", "inference_outputs/gpt-4__swe_bench_lite_demo3_turns.jsonl"),
        ("gpt4__SWE-bench_Lite__default_test_demo4__t-0.00__p-0.95__c-3.00__install-1", "sweap__gpt-4-1106-preview", r"\sweap", "inference_outputs/gpt-4__swe_bench_lite_demo4_turns.jsonl"),
    ]

    headers = (
        ["Method", r"{$\bc{A}$ \up{}}", r"{\ftx \up{}}", r"{\ftp \up{}}", r"{\ptp}"]
        if format == "latex" else
        ["Method", "Applicability", "F2X", "F2P", "P2P"]
    )
    rows = []
    for model, run_id, name, num_turns_file, *args in methods:
        with open(num_turns_file) as f:
            num_turns = [json.loads(l) for l in (f)]
        num_turns_per_instance = {x["instance_id"]: x["num_turns"] for x in num_turns}
        reports = collect_reports(model, run_id, instance_log_path, *args)
        for i in range(1, max_turns+1, max_turns//5):
            reports_i = {k: v for k, v in reports.items() if num_turns_per_instance[k] <= i}
            applied = 100*no_error_count(reports_i)/total_instance_count
            ftp = 100*ftp_count(reports_i)/total_instance_count
            ftx = 100*ftx_count(reports_i)/total_instance_count
            ptp = 100*ptp_count(reports_i)/total_instance_count
            rows.append([f"{name} {i}", applied, ftx, ftp, ptp])
    print(tabulate(rows, headers=headers, tablefmt=format, floatfmt=".1f"))

if __name__ == "__main__":
    fire.Fire(main)