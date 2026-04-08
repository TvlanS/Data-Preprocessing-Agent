"""
core/change_log.py
------------------
Records every operation that was applied (or previewed in dry-run mode).
Saved to change_log.json at the end of the pipeline run.
"""

import json
from datetime import datetime


class ChangeLog:
    def __init__(self):
        self._entries: list[dict] = []

    def record(
        self,
        op_name:  str,
        column:   str,
        params:   dict,
        reason:   str,
        confidence: float,
        dry_run:  bool = False,
        error:    str  = None,
    ):
        self._entries.append({
            "timestamp":  datetime.now().isoformat(timespec="seconds"),
            "op":         op_name,
            "column":     column,
            "params":     params,
            "reason":     reason,
            "confidence": round(confidence, 3),
            "dry_run":    dry_run,
            "error":      error,
        })

    def count(self) -> int:
        return len(self._entries)

    def save(self, path: str = "change_log.json"):
        with open(path, "w") as f:
            json.dump(
                {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "total_ops":    len(self._entries),
                    "entries":      self._entries,
                },
                f,
                indent=2,
            )
        print(f"[changelog] Saved {len(self._entries)} entries to {path}")

    def to_list(self) -> list[dict]:
        return list(self._entries)
