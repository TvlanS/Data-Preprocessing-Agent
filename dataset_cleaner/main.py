"""
Dataset Cleaning Agent
======================
Entry point. Run this to start the full pipeline.

Usage:
    python main.py --input data.csv
    python main.py --input data.csv --dry-run
    python main.py --input data.csv --skip-audit   # if audit_report.json already exists
"""

import argparse
import json
import os
import pandas as pd

from core.auditor import run_audit
from core.deepseek_client import get_cleaning_ops
from core.router import OpRouter
from core.change_log import ChangeLog
from tools.builtin_tools import TOOLS
from review.review_gate import ReviewGate


def load_custom_tools():
    """Load any previously approved custom tools into TOOLS registry."""
    custom_path = "custom_tools.py"
    if not os.path.exists(custom_path):
        return
    import importlib.util
    spec = importlib.util.spec_from_file_location("custom_tools", custom_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    count = 0
    for name, fn in vars(mod).items():
        if callable(fn) and not name.startswith("_"):
            TOOLS[name] = fn
            count += 1
    if count:
        print(f"[startup] Loaded {count} custom tool(s) from custom_tools.py")


def main():
    parser = argparse.ArgumentParser(description="Dataset Cleaning Agent")
    parser.add_argument("--input",       required=True,  help="Path to input dataset")
    parser.add_argument("--output",      default="clean_output.csv", help="Path to cleaned output")
    parser.add_argument("--report",      default="audit_report.json", help="Path to audit report JSON")
    parser.add_argument("--dry-run",     action="store_true", help="Preview changes without writing")
    parser.add_argument("--skip-audit",  action="store_true", help="Skip audit, use existing report")
    parser.add_argument("--min-score",   type=int, default=80, help="Skip columns above this score")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  DATASET CLEANING AGENT")
    print("="*60)

    # ── 1. Load custom tools from previous runs ──────────────────
    load_custom_tools()

    # ── 2. Run audit (or load existing report) ───────────────────
    if args.skip_audit and os.path.exists(args.report):
        print(f"\n[audit] Loading existing report: {args.report}")
        with open(args.report) as f:
            report = json.load(f)
    else:
        print(f"\n[audit] Running DatasetAuditor on: {args.input}")
        report = run_audit(args.input, args.report)

    # ── 3. Load dataset ──────────────────────────────────────────
    ext = os.path.splitext(args.input)[1].lower()
    loaders = {
        ".csv":     pd.read_csv,
        ".xlsx":    pd.read_excel,
        ".xls":     pd.read_excel,
        ".parquet": pd.read_parquet,
        ".json":    pd.read_json,
    }
    if ext not in loaders:
        raise ValueError(f"Unsupported file format: {ext}")
    df = loaders[ext](args.input)
    print(f"[load] Dataset shape: {df.shape}")

    # ── 4. Sort columns by quality score (worst first) ───────────
    columns = sorted(report["columns"], key=lambda c: c["quality_score"])

    # ── 5. Initialise shared components ─────────────────────────
    router    = OpRouter(TOOLS, dry_run=args.dry_run)
    changelog = ChangeLog()
    reviewer  = ReviewGate(TOOLS, changelog, dry_run=args.dry_run)

    # ── 6. Main cleaning loop ────────────────────────────────────
    print(f"\n[pipeline] Processing {len(columns)} columns "
          f"(skipping scores >= {args.min_score})\n")

    for col_report in columns:
        col   = col_report["column"]
        score = col_report["quality_score"]

        if score >= args.min_score:
            print(f"  [skip]  {col:<30} score={score}  (clean)")
            continue

        status = "GOOD" if score >= 80 else ("WARN" if score >= 50 else "BAD")
        print(f"\n  [{status}]  {col}  (score={score})")

        # ── Ask DeepSeek for cleaning ops ────────────────────────
        ops = get_cleaning_ops(col_report, list(TOOLS.keys()))

        if not ops:
            print(f"    -> No ops returned, skipping.")
            continue

        # ── Separate standard ops from flagged ones ───────────────
        standard_ops = [o for o in ops if o["op"] != "flag_for_review"]
        flagged_ops  = [o for o in ops if o["op"] == "flag_for_review"]

        # ── Apply standard ops via router ────────────────────────
        if standard_ops:
            df = router.apply(df, standard_ops, changelog)

        # ── Handle flagged columns through review gate ────────────
        for flag in flagged_ops:
            df = reviewer.handle(df, col_report, flag)

    # ── 7. Save output ───────────────────────────────────────────
    print("\n" + "="*60)
    if args.dry_run:
        print("[dry-run] No files written. Change preview above.")
    else:
        df.to_csv(args.output, index=False)
        changelog.save("change_log.json")
        print(f"[done] Cleaned dataset  -> {args.output}")
        print(f"[done] Change log       -> change_log.json")

    print(f"[done] Total operations applied: {changelog.count()}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
