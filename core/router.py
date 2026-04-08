"""
core/router.py
--------------
Validates and dispatches a list of op dicts to the correct pandas tool function.
Supports dry-run mode (prints what would happen without touching the dataframe).

Op execution order is enforced so that destructive ops (drop_column, drop_duplicates)
always run last within a batch.
"""

import pandas as pd
from core.change_log import ChangeLog

# Ops that must run after all other ops for a column batch
DEFERRED_OPS = {"drop_column", "drop_duplicates"}


class OpRouter:
    def __init__(self, tools: dict, dry_run: bool = False):
        self.tools   = tools
        self.dry_run = dry_run

    def _sort_ops(self, ops: list[dict]) -> list[dict]:
        """Run deferred ops (drop_*) last."""
        normal   = [o for o in ops if o["op"] not in DEFERRED_OPS]
        deferred = [o for o in ops if o["op"] in DEFERRED_OPS]
        return normal + deferred

    def apply(self, df: pd.DataFrame, ops: list[dict], changelog: ChangeLog) -> pd.DataFrame:
        """Apply a list of validated ops to df. Returns modified df."""
        ops = self._sort_ops(ops)

        for op in ops:
            op_name = op["op"]
            column  = op.get("column", "")
            params  = op.get("params", {})
            reason  = op.get("reason", "")
            conf    = op.get("confidence", 1.0)

            if op_name not in self.tools:
                print(f"    [router] Unknown op '{op_name}' — skipped.")
                continue

            if self.dry_run:
                print(f"    [dry-run] Would apply '{op_name}' to '{column}'")
                print(f"              reason: {reason}")
                print(f"              params: {params}")
                print(f"              confidence: {conf:.2f}")
                changelog.record(op_name, column, params, reason, conf, dry_run=True)
                continue

            try:
                before_shape = df.shape
                df = self.tools[op_name](df, column, params)
                after_shape  = df.shape
                print(f"    [apply]  '{op_name}' on '{column}'  "
                      f"shape {before_shape} -> {after_shape}")
                changelog.record(op_name, column, params, reason, conf)
            except Exception as e:
                print(f"    [error]  '{op_name}' on '{column}' failed: {e}")
                changelog.record(op_name, column, params, reason, conf, error=str(e))

        return df
