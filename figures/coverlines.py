"""
Method vs Applicability, FtX, FtP and PtP
setting of directed fuzzing
"""

from tabulate import tabulate
import fire

from figures.util import *

def main(instance_log_path: str = "./run_instance_swt_logs", total_instance_count: int = 279, format: str = "github"):
    instance_log_path = Path(instance_log_path)
    if not instance_log_path.exists():
        raise FileNotFoundError(f"Instance log directory not found at {instance_log_path}")
    methods = [
        ("gold", "validate-gold", r"Gold"),
        ("gpt-4-1106-preview", "gpt-4-1106-preview__swt_bench_linecover_oracle__seed=0,temperature=0__test", r"\zsp"),
        ("gpt-4-1106-preview", "gpt-4-1106-preview__swt_bench_linecover_oracle__seed=0,temperature=0.7,n=5__test._0", r"\pak", "p@k", [0,1,2,3,4]),
        ("gpt4__SWE-bench_Lite__default_test_demo6__t-0.00__p-0.95__s-0__c-3.00__install-1", "swe-agent-demo6__swt_bench_lite_coverlines__test", r"\sweap"),
    ]

    headers = (
        ["Method", r"{$\bc{A}}$ \up{}}", r"{\ftx \up{}}", r"{\ftp \up{}}", r"{\ptp}", r"{$\dc^{\text{all}}$ }", r"{$\dc^{\ftp}$}", r"{$\dc^{\neg(\ftp)}$}"]
        if format == "latex" else
        ["Method", "Applicability", "F2X", "F2P", "P2P", "Coverage", "Resolved Coverage", "Unresolved Coverage"]
    )
    rows = []
    for model, run_id, name, *args in methods:
        reports = collect_reports(model, run_id, instance_log_path, *args)
        applied = 100*no_error_count(reports)/total_instance_count
        ftp = 100*ftp_count(reports)/total_instance_count
        ftx = 100*ftx_count(reports)/total_instance_count
        ptp = 100*ptp_count(reports)/total_instance_count
        resolved_reports, unresolved_reports = filtered_by_resolved(reports)
        total_coverage_delta = 100*avg_coverage_delta(reports)
        resolved_coverage_delta = 100*avg_coverage_delta(resolved_reports)
        unresolved_coverage_delta = 100*avg_coverage_delta(unresolved_reports)
        rows.append([name, applied, ftx, ftp, ptp, total_coverage_delta, resolved_coverage_delta, unresolved_coverage_delta])
    print(tabulate(rows, headers=headers, tablefmt=format, floatfmt=".1f"))

if __name__ == "__main__":
    fire.Fire(main)