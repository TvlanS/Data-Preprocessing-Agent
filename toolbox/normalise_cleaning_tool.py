"""
step_1_cleaning_tool.py
═══════════════════════════════════════════════════════════════════════════════
Step 1 of the Data Cleaning Agent pipeline.

Responsibilities
────────────────
  Profile #1   — ydata-profiling snapshot BEFORE any changes
  Normalize    — strip whitespace + lowercase headers & categorical values
  Deduplicate  — remove exact duplicate rows
  Sparse rows  — drop rows exceeding row_threshold % empty
  Sparse cols  — drop columns exceeding col_threshold % empty
  Profile #2   — ydata-profiling snapshot AFTER cleaning
  Log          — write checkpoint events to agent_log.json after each step

Resumable
─────────
  If the script is interrupted at any point, re-running it will skip every
  sub-step already present in agent_log.json and continue from where it left
  off. The cleaned dataset is always loaded from disk on resume rather than
  re-computed, ensuring reproducibility.

Output folder layout (mirrors storage framework)
─────────────────────────────────────────────────
  runs/{dataset}_{YYYYMMDD}_{HHMMSS}/
  ├── agent_log.json
  ├── pipeline_config.json
  ├── input/
  │   └── raw_dataset.*              ← original file, never modified
  ├── profiling/
  │   ├── 01_before_cleaning/
  │   │   ├── report.html
  │   │   └── summary.json
  │   └── 02_after_cleaning/
  │       ├── report.html
  │       └── summary.json
  └── step_1_cleaning/
      ├── dataset.csv
      └── summary.json
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from pyprojroot import here
from ydata_profiling import ProfileReport

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_SEP = "═" * 60


# ══════════════════════════════════════════════════════════════════════════════
# Run setup helper
# ══════════════════════════════════════════════════════════════════════════════

def setup_run(
    dataset_path:      str | Path,
    row_threshold:     float,
    col_threshold:     float,
    minimal_profiling: bool = False,
) -> tuple[Path, dict]:
    """
    Create the run folder structure, write ``pipeline_config.json``, and
    register the run in ``data/pipeline_runs.json``.

    Parameters
    ----------
    dataset_path:
        Path to the raw input file (.csv, .xlsx, .xls).
    row_threshold:
        Drop rows with more than this % of empty cells (e.g. ``70.0``).
    col_threshold:
        Drop columns with more than this % of empty cells (e.g. ``70.0``).
    minimal_profiling:
        Pass ``True`` to run a lighter-weight profiling scan (skips
        correlation analysis). Recommended for datasets > 100k rows.

    Returns
    -------
    (run_dir, config)
        ``run_dir``  – Path to the newly created run folder.
        ``config``   – The pipeline_config dict that was written to disk.
    """
    dataset_path = Path(dataset_path)
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id       = f"{dataset_path.stem}_{timestamp}"
    run_dir      = here("data") / "runs" / run_id

    # Create the required sub-directories up front
    for sub in ["input", "profiling/01_before_cleaning",
                "profiling/02_after_cleaning", "step_1_cleaning"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    config = {
        "run_id":            run_id,
        "source_file":       str(dataset_path),
        "row_threshold":     row_threshold,
        "col_threshold":     col_threshold,
        "minimal_profiling": minimal_profiling,
        "created_at":        datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(run_dir / "pipeline_config.json", config)

    # Register in the top-level pipeline_runs.json index
    runs_index_path = here("data") / "pipeline_runs.json"
    runs_index = _read_json(runs_index_path) if runs_index_path.exists() else {}
    runs_index.setdefault("runs", []).append({
        "run_id":      run_id,
        "started_at":  config["created_at"],
        "status":      "in_progress",
        "source_file": str(dataset_path),
        "run_path":    str(run_dir),
    })
    _write_json(runs_index_path, runs_index)

    logger.info(f"Run created: {run_dir}")
    return run_dir, config


# ══════════════════════════════════════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════════════════════════════════════

def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_dataset(path: str | Path) -> pd.DataFrame:
    """Load CSV or Excel into a DataFrame, auto-detecting the extension."""
    path = Path(path)
    ext  = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in (".xlsx", ".xlsm"):
        return pd.read_excel(path, engine="openpyxl")
    elif ext == ".xls":
        return pd.read_excel(path, engine="xlrd")
    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            "Supported: .csv, .xlsx, .xlsm, .xls"
        )


# ══════════════════════════════════════════════════════════════════════════════
# DataCleaningTool
# ══════════════════════════════════════════════════════════════════════════════

class DataCleaningTool:
    """
    Step 1 of the Data Cleaning Agent pipeline.

    Designed to be called by the agent after ``setup_run()`` has created
    the run folder. Pass the raw DataFrame to ``run()``; the tool handles
    everything else, including resumability via ``agent_log.json``.

    Example
    -------
    >>> run_dir, config = setup_run("data.xlsx", row_threshold=70, col_threshold=70)
    >>> df = _load_dataset("data.xlsx")
    >>> tool = DataCleaningTool(run_dir, row_threshold=70, col_threshold=70)
    >>> df_clean = tool.run(df)
    """

    # Log checkpoint keys
    STEP_PROFILING_1 = "profiling_1_done"
    STEP_CLEANING    = "cleaning_done"
    STEP_PROFILING_2 = "profiling_2_done"

    def __init__(
        self,
        run_dir:           str | Path,
        row_threshold:     float,
        col_threshold:     float,
        minimal_profiling: bool = False,
    ) -> None:
        self.run_dir           = Path(run_dir)
        self.row_threshold     = row_threshold
        self.col_threshold     = col_threshold
        self.minimal_profiling = minimal_profiling

        # ── Derived paths ──────────────────────────────────────────────────────
        self.log_path      = self.run_dir / "agent_log.json"
        self.profile_1_dir = self.run_dir / "profiling" / "01_before_cleaning"
        self.profile_2_dir = self.run_dir / "profiling" / "02_after_cleaning"
        self.step_dir      = self.run_dir / "step_1_cleaning"

        # Ensure directories exist (safe to call even on resume)
        for d in [self.profile_1_dir, self.profile_2_dir, self.step_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Log helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _read_log(self) -> dict:
        """Return agent_log.json content, or an empty scaffold if not found."""
        if self.log_path.exists():
            return _read_json(self.log_path)
        return {
            "completed_steps": [],
            "current_step":    None,
            "user_gates":      {},
        }

    def _step_done(self, step_key: str) -> bool:
        """Return True if *step_key* is already present in the log."""
        log = self._read_log()
        return any(s["step"] == step_key for s in log.get("completed_steps", []))

    def _log_step(
        self,
        step_key:    str,
        output_path: str = "",
        extra:       dict | None = None,
    ) -> None:
        """
        Append a completed-step entry to agent_log.json.
        Writes atomically (read → mutate → write) to avoid partial writes.
        """
        log = self._read_log()
        entry: dict = {
            "step":         step_key,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "output":       output_path,
        }
        if extra:
            entry.update(extra)
        log.setdefault("completed_steps", []).append(entry)
        log["current_step"] = step_key
        log["last_updated"] = datetime.now().isoformat(timespec="seconds")
        _write_json(self.log_path, log)
        logger.info(f"[log] {step_key}")

    # ══════════════════════════════════════════════════════════════════════════
    # Profiling
    # ══════════════════════════════════════════════════════════════════════════

    def _run_profiling(
        self,
        df:      pd.DataFrame,
        out_dir: Path,
        label:   str,
    ) -> dict:
        """
        Run ydata-profiling on *df*, save ``report.html`` and
        ``summary.json`` to *out_dir*.

        Returns the lightweight summary dict.
        """
        logger.info(f"Profiling [{label}] — {len(df):,} rows × {len(df.columns)} cols …")
        title   = f"Data Profile — {label.replace('_', ' ').title()}"
        profile = ProfileReport(
            df,
            title=title,
            minimal=self.minimal_profiling,
            progress_bar=False,
        )
        profile.to_file(out_dir / "report.html")
        profile.to_file(out_dir / "report_json.json")

        desc  = profile.get_description()
        table = desc.table

        summary = {
            "profile_label":  label,
            "generated_at":   datetime.now().isoformat(timespec="seconds"),
            "rows":           int(table["n"]),
            "cols":           int(table["n_var"]),
            "null_pct":       round(float(table["p_cells_missing"]) * 100, 2),
            "duplicate_rows": int(table["n_duplicates"]),
            "dtypes": {
                "numeric":     int(table.get("n_numeric",     0)),
                "categorical": int(table.get("n_categorical", 0)),
                "datetime":    int(table.get("n_date",        0)),
            },
        }
        _write_json(out_dir / "summary.json", summary)
        logger.info(f"Profiling saved → {out_dir}")
        return summary

    # ══════════════════════════════════════════════════════════════════════════
    # Internal utilities
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _build_empty_mask(df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a boolean DataFrame that is True wherever a cell is:
          • NaN / None
          • An empty string or a whitespace-only string

        This is the single definition of "empty" used across every cleaning step.
        """
        nan_mask = df.isna()

        # Only inspect object-typed columns for empty strings; numeric/date
        # columns can only be NaN, never an empty string.
        str_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        for col in df.select_dtypes(include="object").columns:
            # astype(str) turns NaN → "nan", which strip() ≠ "" → stays False.
            # Genuine empty strings and whitespace-only values → True.
            str_mask[col] = df[col].astype(str).str.strip() == ""

        return nan_mask | str_mask

    # ══════════════════════════════════════════════════════════════════════════
    # Cleaning steps (each returns (df, change_info))
    # ══════════════════════════════════════════════════════════════════════════

    def _normalize_headers(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict]:
        """
        Strip leading/trailing whitespace and lowercase all column names.
        Spaces within a name are replaced with underscores.

        Returns (df, mapping) where *mapping* records every name that changed.
        """
        mapping  : dict[str, str] = {}
        new_cols : list[str]      = []

        for col in df.columns:
            cleaned = str(col).strip().lower().replace(" ", "_")
            if cleaned != str(col):
                mapping[str(col)] = cleaned
            new_cols.append(cleaned)

        df.columns = new_cols

        if mapping:
            logger.info(f"  Headers — normalised {len(mapping)}: {mapping}")
        else:
            logger.info("  Headers — no changes needed")

        return df, mapping

    def _normalize_categoricals(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Strip whitespace and lowercase every string value in object / category
        columns. Non-string cells (NaN, numbers stored as object) are untouched.

        Returns (df, list_of_changed_column_names).
        """
        changed: list[str] = []

        for col in df.select_dtypes(include=["object", "category"]).columns:
            before = df[col].copy()
            df[col] = df[col].map(
                lambda x: x.strip().lower() if isinstance(x, str) else x
            )
            if not df[col].equals(before):
                changed.append(col)

        if changed:
            logger.info(
                f"  Categoricals — normalised values in {len(changed)} cols: {changed}"
            )
        else:
            logger.info("  Categoricals — no changes needed")

        return df, changed

    def _drop_duplicates(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, int]:
        """
        Remove exact duplicate rows (all columns must match).
        Returns (df, n_removed).
        """
        before = len(df)
        df     = df.drop_duplicates().reset_index(drop=True)
        n      = before - len(df)
        logger.info(f"  Duplicates removed: {n:,}")
        return df, n

    def _drop_sparse_rows(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, int]:
        """
        Drop rows where the fraction of empty cells *exceeds* row_threshold %.
        Returns (df, n_removed).
        """
        empty_mask    = self._build_empty_mask(df)
        pct_by_row    = empty_mask.sum(axis=1) / len(df.columns) * 100
        drop_mask     = pct_by_row > self.row_threshold
        n             = int(drop_mask.sum())
        df            = df[~drop_mask].reset_index(drop=True)
        logger.info(
            f"  Sparse rows dropped (>{self.row_threshold}% empty): {n:,}"
        )
        return df, n

    def _drop_sparse_columns(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Drop columns where the fraction of empty cells *exceeds* col_threshold %.
        Returns (df, list_of_dropped_column_names).
        """
        empty_mask = self._build_empty_mask(df)
        pct_by_col = empty_mask.sum(axis=0) / len(df) * 100
        to_drop    = pct_by_col[pct_by_col > self.col_threshold].index.tolist()

        for col in to_drop:
            logger.info(
                f"    Dropping column '{col}': {pct_by_col[col]:.1f}% empty"
            )

        if to_drop:
            df = df.drop(columns=to_drop)
            logger.info(
                f"  Sparse columns dropped (>{self.col_threshold}% empty): "
                f"{len(to_drop)}"
            )
        else:
            logger.info("  Sparse columns — none to drop")

        return df, to_drop

    # ══════════════════════════════════════════════════════════════════════════
    # Diagnostic helpers  (non-destructive — do not modify df)
    # ══════════════════════════════════════════════════════════════════════════

    def display_stats(self, df: pd.DataFrame) -> dict:
        """
        Print a human-readable diagnostics table for the given DataFrame
        without modifying it.

        Use this before *and* after ``run()`` to get a quick visual diff.
        Returns the same stats as a dict for programmatic use.
        """
        empty_mask    = self._build_empty_mask(df)
        pct_by_row    = empty_mask.sum(axis=1) / len(df.columns) * 100
        n_sparse_rows = int((pct_by_row > self.row_threshold).sum())
        pct_by_col    = empty_mask.sum(axis=0) / len(df) * 100
        sparse_cols   = pct_by_col[pct_by_col > self.col_threshold].index.tolist()

        n_dup   = int(df.duplicated().sum())
        dup_pct = round(n_dup / len(df) * 100, 2) if len(df) else 0.0
        row_pct = round(n_sparse_rows / len(df) * 100, 2) if len(df) else 0.0

        print()
        print("─" * 60)
        print(f"  Diagnostics   {len(df):>10,} rows  ×  {len(df.columns)} cols")
        print("─" * 60)
        print(f"  Duplicate rows            : {n_dup:>8,}  ({dup_pct:.2f}%)")
        print(
            f"  Sparse rows (>{self.row_threshold}% empty)  : "
            f"{n_sparse_rows:>8,}  ({row_pct:.2f}%)"
        )
        print(f"  Sparse cols (>{self.col_threshold}% empty)  : {len(sparse_cols):>8}")
        if sparse_cols:
            print(f"    → {sparse_cols}")
        print("─" * 60)
        print()

        return {
            "shape":            df.shape,
            "duplicate_count":  n_dup,
            "duplicate_pct":    dup_pct,
            "sparse_row_count": n_sparse_rows,
            "sparse_row_pct":   row_pct,
            "sparse_cols":      sparse_cols,
        }

    def find_unique_categorical_columns(
        self, df: pd.DataFrame, threshold: float = 10.0
    ) -> list[str]:
        """
        Identify categorical columns that look like discrete identifiers —
        columns where even the single most-common value appears in fewer than
        *threshold* % of non-null rows.

        These are candidates to exclude from value normalisation or outlier
        analysis (e.g. patient IDs, record numbers, free-text fields).

        Parameters
        ----------
        threshold:
            Dominance ceiling in percent. Default 10 %.

        Returns
        -------
        List of column names that appear to be discrete identifiers.
        """
        discrete: list[str] = []

        for col in df.select_dtypes(include=["object", "category"]).columns:
            non_null = df[col].count()
            if non_null == 0:
                continue
            top_pct = df[col].value_counts().iloc[0] / non_null * 100
            if top_pct < threshold:
                discrete.append(col)

        if discrete:
            logger.info(f"Discrete / identifier columns found: {discrete}")
        else:
            logger.info("No discrete identifier columns found")

        return discrete

    # ══════════════════════════════════════════════════════════════════════════
    # Main orchestrator
    # ══════════════════════════════════════════════════════════════════════════

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the full Step-1 pipeline.

        Each sub-step checks ``agent_log.json`` before running:
          • If the step is already logged → skip and move on.
          • If the cleaning step is logged but profiling-2 is not → reload
            the saved CSV from disk so the profiling runs on the correct data.

        Parameters
        ----------
        df:
            The raw DataFrame loaded from ``input/raw_dataset.*``.
            Must not be pre-filtered — this is the very first step.

        Returns
        -------
        pd.DataFrame
            The cleaned DataFrame (also saved to ``step_1_cleaning/dataset.csv``).
        """
        logger.info(_SEP)
        logger.info("  Step 1 — Data Cleaning Tool")
        logger.info(_SEP)

        # ── Sub-step 1: Profile before cleaning ───────────────────────────────
        if not self._step_done(self.STEP_PROFILING_1):
            s1 = self._run_profiling(df, self.profile_1_dir, "before_cleaning")
            self._log_step(
                self.STEP_PROFILING_1,
                output_path=str(self.profile_1_dir),
                extra={
                    "rows":     s1["rows"],
                    "cols":     s1["cols"],
                    "null_pct": s1["null_pct"],
                },
            )
        else:
            logger.info(f"[skip] {self.STEP_PROFILING_1} — already done")

        # ── Sub-step 2: Cleaning ───────────────────────────────────────────────
        if not self._step_done(self.STEP_CLEANING):
            rows_in = len(df)
            cols_in = len(df.columns)

            logger.info("Cleaning …")
            df, header_map    = self._normalize_headers(df)
            df, cat_cols      = self._normalize_categoricals(df)
            df, dupes_removed = self._drop_duplicates(df)
            df, rows_removed  = self._drop_sparse_rows(df)
            df, cols_removed  = self._drop_sparse_columns(df)

            # Persist cleaned dataset
            csv_path = self.step_dir / "dataset.csv"
            df.to_csv(csv_path, index=False)

            # Persist human-readable cleaning summary
            cleaning_summary = {
                "completed_at":           datetime.now().isoformat(timespec="seconds"),
                "rows_before":            rows_in,
                "rows_after":             len(df),
                "cols_before":            cols_in,
                "cols_after":             len(df.columns),
                "duplicates_removed":     dupes_removed,
                "rows_dropped_sparse":    rows_removed,
                "cols_dropped_sparse":    cols_removed,
                "headers_normalised":     header_map,
                "categorical_normalised": cat_cols,
            }
            _write_json(self.step_dir / "summary.json", cleaning_summary)

            self._log_step(
                self.STEP_CLEANING,
                output_path=str(csv_path),
                extra={
                    "rows_removed": dupes_removed + rows_removed,
                    "cols_removed": len(cols_removed),
                },
            )

        else:
            # Resume path: reload the already-cleaned dataset from disk so that
            # profile #2 runs on the exact same data as the original clean pass.
            csv_path = self.step_dir / "dataset.csv"
            logger.info(
                f"[skip] {self.STEP_CLEANING} — loading from disk: {csv_path}"
            )
            df = pd.read_csv(csv_path)

        # ── Sub-step 3: Profile after cleaning ────────────────────────────────
        if not self._step_done(self.STEP_PROFILING_2):
            s2 = self._run_profiling(df, self.profile_2_dir, "after_cleaning")
            self._log_step(
                self.STEP_PROFILING_2,
                output_path=str(self.profile_2_dir),
                extra={
                    "rows":     s2["rows"],
                    "cols":     s2["cols"],
                    "null_pct": s2["null_pct"],
                },
            )
        else:
            logger.info(f"[skip] {self.STEP_PROFILING_2} — already done")

        logger.info("Step 1 complete ✓")
        logger.info(_SEP)
        return df


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Configuration ─────────────────────────────────────────────────────────
    DATASET_PATH      = r"C:\Users\tvlan\Documents\Data Mining\1.0 Assignment\1.0 Python\Data\tb_syth_data_basic_V5.csv"
    ROW_THRESHOLD     = 70.0   # Drop rows  with > 70 % empty cells
    COL_THRESHOLD     = 70.0   # Drop columns with > 70 % empty cells
    MINIMAL_PROFILING = False  # True = faster scan, skips correlation analysis

    dataset_path = Path(DATASET_PATH)

    # ── 1. Create run folder + config ─────────────────────────────────────────
    run_dir, config = setup_run(
        dataset_path,
        row_threshold=ROW_THRESHOLD,
        col_threshold=COL_THRESHOLD,
        minimal_profiling=MINIMAL_PROFILING,
    )

    # ── 2. Load raw dataset ───────────────────────────────────────────────────
    logger.info(f"Loading: {dataset_path}")
    df_raw = _load_dataset(dataset_path)
    logger.info(f"Loaded  {len(df_raw):,} rows × {len(df_raw.columns)} cols")

    # ── 3. Save an untouched copy to input/ ───────────────────────────────────
    input_copy = run_dir / "input" / dataset_path.name
    if not input_copy.exists():
        shutil.copy2(dataset_path, input_copy)
        logger.info(f"Raw file archived → {input_copy}")

    # ── 4. Inspect before cleaning ────────────────────────────────────────────
    tool = DataCleaningTool(run_dir, ROW_THRESHOLD, COL_THRESHOLD, MINIMAL_PROFILING)
    print("── BEFORE ──────────────────────────────────────────────────")
    tool.display_stats(df_raw)

    # Optional: flag discrete identifier columns so you know what to expect
    tool.find_unique_categorical_columns(df_raw)

    # ── 5. Run Step 1 ─────────────────────────────────────────────────────────
    df_clean = tool.run(df_raw)

    # ── 6. Inspect after cleaning ─────────────────────────────────────────────
    print("── AFTER ───────────────────────────────────────────────────")
    tool.display_stats(df_clean)

    print(f"\nOutputs written to: {run_dir}")
    print(f"  Cleaned dataset : {run_dir / 'step_1_cleaning' / 'dataset.csv'}")
    print(f"  Profile #1      : {run_dir / 'profiling' / '01_before_cleaning' / 'report.html'}")
    print(f"  Profile #2      : {run_dir / 'profiling' / '02_after_cleaning' / 'report.html'}")
    print(f"  Agent log       : {run_dir / 'agent_log.json'}")
