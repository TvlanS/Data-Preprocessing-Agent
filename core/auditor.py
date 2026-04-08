"""
core/auditor.py
---------------
Thin wrapper around the DatasetAuditor class from dataset_audit_report.py.
Runs the audit and returns the parsed JSON report as a dict.
"""

import json
import os
import sys


def run_audit(input_path: str, report_path: str = "audit_report.json") -> dict:
    """
    Run the DatasetAuditor pipeline on input_path.
    Returns the parsed audit report as a Python dict.

    Expects dataset_audit_report.py to be in the project root or sys.path.
    """
    try:
        from dataset_audit_report import DatasetAuditor
    except ImportError:
        print(
            "\n[error] Could not import DatasetAuditor.\n"
            "        Place dataset_audit_report.py in the project root.\n"
        )
        sys.exit(1)

    auditor = DatasetAuditor(input_path)
    auditor.run()

    # DatasetAuditor saves to audit_report.json by default.
    # If the caller specified a different path, move it.
    default_path = "audit_report.json"
    if report_path != default_path and os.path.exists(default_path):
        os.rename(default_path, report_path)

    if not os.path.exists(report_path):
        raise FileNotFoundError(
            f"Audit report not found at {report_path}. "
            "Check that DatasetAuditor.run() completed successfully."
        )

    with open(report_path) as f:
        return json.load(f)
