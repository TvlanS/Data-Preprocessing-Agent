"""
Dataset Cleaning Agent - Audit Report Generator
================================================
Usage (in Python):

    from dataset_audit_report import DatasetAuditor

    auditor = DatasetAuditor("C:/Users/tvlan/Downloads/heasrtatatck/heart_attack_prediction_dataset.csv")
    auditor.run()

Optional — change output folder or format:
    auditor = DatasetAuditor(
        input_path="C:/Users/tvlan/Downloads/heasrtatatck/heart_attack_prediction_dataset.csv",
        output_dir="my_reports",
        fmt="both"
    )
    auditor.run()

Dependencies:
    pip install pandas numpy matplotlib seaborn scipy rapidfuzz scikit-learn openpyxl
"""

import json
import os
import re
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Optional dependencies ────────────────────────────────────────────────────
try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

CARDINALITY_THRESHOLD        = 0.05
HIGH_MISSING_THRESHOLD       = 0.40
DISTORTED_CATEGORY_THRESHOLD = 0.80
MAX_CATEGORIES_TO_SHOW       = 30
RANDOM_SAMPLE_SIZE           = 5
OUTLIER_Z_THRESHOLD          = 3.0
SPLIT_DELIMITERS             = [",", ";", "|", "/", " - ", " & "]
SEMANTIC_PATTERNS = {
    "email":   r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "phone":   r"(\+?\d[\d\s\-().]{7,}\d)",
    "url":     r"https?://\S+",
    "date":    r"\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b",
    "zipcode": r"\b\d{5}(?:-\d{4})?\b",
}


# ════════════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ════════════════════════════════════════════════════════════════════════════

class DatasetAuditor:
    """
    Audit a dataset and generate a cleaning report.

    Parameters
    ----------
    input_path : str
        Path to your dataset file (.csv, .xlsx, .xls, .parquet, .json)
    output_dir : str
        Folder where reports + boxplots will be saved. Default: "audit_output"
    fmt : str
        Output format: "json", "markdown", or "both". Default: "both"
    """

    def __init__(self, input_path: str, output_dir: str = "audit_output", fmt: str = "both"):
        self.input_path  = input_path
        self.base_output_dir  = output_dir
        self.fmt         = fmt
        self.dataset_name = Path(input_path).stem
        self.timestamp = datetime.now().strftime("%m_%d_%Y_%I_%M_%p")
        self.run_folder_name = f"{self.dataset_name}_a{self.timestamp}"
        self.output_dir = os.path.join(self.base_output_dir, self.run_folder_name)
        self.boxplot_dir = os.path.join(self.output_dir, "boxplots")
        self.report      = {}
        self.df          = None

    # ── Public entry point ───────────────────────────────────────────────────

    def run(self):
        """Load data, run all audits, save reports, print summary."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.boxplot_dir, exist_ok=True)

        print(f" Loading: {self.input_path}")
        self.df = self._load(self.input_path)
        print(f"   Shape  : {self.df.shape}")

        print("Running audit...")
        self._build_report()
        self._save_reports()
        self._print_summary()

    # ── Loader ───────────────────────────────────────────────────────────────

    def _load(self, path: str) -> pd.DataFrame:
        ext = Path(path).suffix.lower()
        loaders = {
            ".csv":     pd.read_csv,
            ".xlsx":    pd.read_excel,
            ".xls":     pd.read_excel,
            ".parquet": pd.read_parquet,
            ".json":    pd.read_json,
        }
        if ext not in loaders:
            raise ValueError(f"Unsupported file type: {ext}")
        return loaders[ext](path)

    # ── Report builder ───────────────────────────────────────────────────────

    def _build_report(self):
        
        self.report = {
            "meta": {
                "generated_at":  datetime.now().isoformat(),
                "source_file":   self.input_path,
                "total_rows":    len(self.df),
                "total_columns": len(self.df.columns),
            },
            "column_header_audit": {},
            "duplicate_audit":     {},
            "columns": {},  # Dictionary with column names as keys
            "summary":             {},
        }

        # 1. Normalise headers
        self.df, changes = self._normalize_column_names(self.df)
        self.report["column_header_audit"] = {
            "columns_renamed": len(changes),
            "changes":         changes,
            "final_columns":   list(self.df.columns),
            "recommendation": (
                f"{len(changes)} column(s) normalised to UPPERCASE."
                if changes else "All headers already consistent (UPPERCASE)."
            ),
        }

        # 2. Duplicates
        self.report["duplicate_audit"] = self._find_duplicates(self.df)

        # 3. Per-column - use dictionary with column name as key
        for col in self.df.columns:
            col_type = self._infer_type(self.df[col])
            if col_type == "CATEGORICAL":
                audit = self._audit_categorical(col, self.df[col])
            else:
                audit = self._audit_numerical(col, self.df[col])
            self.report["columns"][col] = audit  # Column name is the key

        # 4. Summary - FIXED: use column names from keys, not from c["column"]
        scores = [c["quality_score"] for c in self.report["columns"].values()]
        self.report["summary"] = {
            "overall_quality_score": round(float(np.mean(scores)), 1) if scores else 0,
            "columns_with_missing":  sum(1 for c in self.report["columns"].values() if c["missing_count"] > 0),
            # FIXED: iterate over items() to get both key (col_name) and value (c)
            "drop_candidates":       [col_name for col_name, c in self.report["columns"].items() 
                                    if c["missing_pct"] >= HIGH_MISSING_THRESHOLD * 100],
            "split_candidates":      [col_name for col_name, c in self.report["columns"].items() 
                                    if c.get("split_analysis", {}).get("should_split")],
            "categorical_columns":   sum(1 for c in self.report["columns"].values() if c["type"] == "CATEGORICAL"),
            "numerical_columns":     sum(1 for c in self.report["columns"].values() if c["type"] == "NUMERICAL"),
            "total_recommendations": sum(len(c["recommendations"]) for c in self.report["columns"].values()),
        }

    # ── Save reports ─────────────────────────────────────────────────────────

    def _save_reports(self):

        base = os.path.join(self.output_dir, "audit_report")

        if self.fmt in ("json", "both"):
            path = base + ".json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.report, f, indent=2, default=str)
            print(f"JSON saved     : {path}")

        if self.fmt in ("markdown", "both"):
            path = base + ".md"
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._render_markdown())
            print(f" Markdown saved : {path}")

        clean_path = base + "_normalised_headers.csv"
        self.df.to_csv(clean_path, index=False)
        print(f"Clean CSV saved: {clean_path}")

    # ── Print summary ────────────────────────────────────────────────────────

    def _print_summary(self):
        s = self.report["summary"]
        d = self.report["duplicate_audit"]
        print("\n" + "=" * 60)
        print(f"  Overall Quality Score : {s['overall_quality_score']} / 100")
        print(f"  Columns with missing  : {s['columns_with_missing']}")
        print(f"  Drop candidates       : {', '.join(s['drop_candidates']) or 'None'}")
        print(f"  Split candidates      : {', '.join(s['split_candidates']) or 'None'}")
        print(f"  Duplicate rows        : {d['total_duplicate_rows']}")
        print("=" * 60)
    # ════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ════════════════════════════════════════════════════════════════════════

    def _normalize_column_names(self, df):
        changes, rename_map = [], {}
        for col in df.columns:
            new = col.strip().upper()
            if new != col:
                changes.append({"original": col, "normalized": new})
                rename_map[col] = new
        return df.rename(columns=rename_map), changes

    def _find_duplicates(self, df):
        mask  = df.duplicated()
        count = int(mask.sum())
        return {
            "total_duplicate_rows":      count,
            "percentage":                round(count / len(df) * 100, 2),
            "recommendation":            "DROP duplicates before modelling." if count > 0 else "No action needed.",
            "example_duplicate_indices": df[mask].head(5).index.tolist(),
        }

    def _infer_type(self, series):
        if pd.api.types.is_numeric_dtype(series):
            n_unique = series.nunique(dropna=True)
            if n_unique / max(len(series), 1) < CARDINALITY_THRESHOLD or n_unique <= 10:
                return "CATEGORICAL"
            return "NUMERICAL"
        return "CATEGORICAL"

    def _detect_split(self, series):
        sample = series.dropna().astype(str).head(200)
        for delim in SPLIT_DELIMITERS:
            hit = sample.str.contains(re.escape(delim), regex=False).mean()
            if hit > 0.5:
                examples = sample[sample.str.contains(re.escape(delim), regex=False)].head(3).tolist()
                return {"should_split": True, "delimiter": delim,
                        "hit_rate_pct": round(hit * 100, 1), "examples": examples}
        return {"should_split": False}

    def _detect_semantic(self, series):
        sample = series.dropna().astype(str).head(200)
        for name, pattern in SEMANTIC_PATTERNS.items():
            if sample.str.match(pattern).mean() > 0.5:
                return name
        return None

    def _fuzzy_candidates(self, values):
        if not HAS_RAPIDFUZZ or len(values) < 2:
            return []
        seen, out = set(), []
        for i, v1 in enumerate(values):
            for v2 in values[i + 1:]:
                pair = tuple(sorted([v1, v2]))
                if pair in seen:
                    continue
                seen.add(pair)
                score = fuzz.ratio(v1.lower(), v2.lower())
                if 70 <= score < 100:
                    out.append({"value_a": v1, "value_b": v2, "similarity_pct": score})
        return sorted(out, key=lambda x: -x["similarity_pct"])[:10]

    def _embed_candidates(self, values):
        if not HAS_SKLEARN or len(values) < 2:
            return []
        try:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
            sim = cosine_similarity(vec.fit_transform(values))
            out = []
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    if sim[i, j] > 0.6:
                        out.append({"value_a": values[i], "value_b": values[j],
                                    "cosine_similarity": round(float(sim[i, j]), 3)})
            return sorted(out, key=lambda x: -x["cosine_similarity"])[:10]
        except Exception:
            return []

    def _quality_score(self, missing_pct, is_distorted, has_outliers, should_split):
        score = 100 - min(missing_pct, 100) * 0.5
        if is_distorted:  score -= 15
        if has_outliers:  score -= 10
        if should_split:  score -= 10
        return max(0, int(round(score)))

    def _detect_outliers(self, series):
        clean = series.dropna()
        if len(clean) < 4:
            return {"method": "insufficient_data", "outlier_count": 0, "outlier_pct": 0}
        skew = float(clean.skew())
        if abs(skew) > 1:
            q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
            iqr    = q3 - q1
            mask   = (clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)
            method = "IQR"
            bounds = {"lower": round(float(q1 - 1.5 * iqr), 4),
                      "upper": round(float(q3 + 1.5 * iqr), 4)}
        else:
            z      = np.abs(stats.zscore(clean))
            mask   = z > OUTLIER_Z_THRESHOLD
            method = "Z-score"
            bounds = {"threshold": OUTLIER_Z_THRESHOLD}
        count = int(mask.sum())
        return {
            "method":         method,
            "skewness":       round(skew, 3),
            "outlier_count":  count,
            "outlier_pct":    round(count / len(clean) * 100, 2),
            "bounds":         bounds,
            "recommendation": ("Investigate outliers — consider capping/winsorizing."
                               if count > 0 else "No significant outliers detected."),
        }

    def _descriptive_stats(self, series):
        d = series.describe()
        return {
            "count":    int(d["count"]),
            "mean":     round(float(d["mean"]), 4),
            "std":      round(float(d["std"]),  4),
            "min":      round(float(d["min"]),  4),
            "p25":      round(float(d["25%"]),  4),
            "median":   round(float(d["50%"]),  4),
            "p75":      round(float(d["75%"]),  4),
            "max":      round(float(d["max"]),  4),
            "skewness": round(float(series.skew()),     3),
            "kurtosis": round(float(series.kurtosis()), 3),
        }

    def _make_boxplot(self, series, col_name):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0f0f0f")
        fig.suptitle(f"Distribution - {col_name}", color="#e0e0e0",
                     fontsize=13, fontweight="bold", y=1.01)
        clean = series.dropna()

        ax1 = axes[0]
        ax1.set_facecolor("#1a1a1a")
        ax1.boxplot(clean, vert=True, patch_artist=True,
                    medianprops=dict(color="#ff6b6b", linewidth=2),
                    boxprops=dict(facecolor="#2a5298", alpha=0.8),
                    whiskerprops=dict(color="#aaaaaa"),
                    capprops=dict(color="#aaaaaa"),
                    flierprops=dict(marker="o", color="#ff6b6b", alpha=0.5, markersize=4))
        ax1.set_title("Box Plot", color="#cccccc", fontsize=10)
        ax1.tick_params(colors="#aaaaaa")
        for sp in ax1.spines.values():
            sp.set_edgecolor("#333333")

        ax2 = axes[1]
        ax2.set_facecolor("#1a1a1a")
        ax2.hist(clean, bins=30, color="#2a5298", alpha=0.7, edgecolor="#1a1a1a", density=True)
        try:
            kde     = stats.gaussian_kde(clean)
            x_range = np.linspace(clean.min(), clean.max(), 300)
            ax2.plot(x_range, kde(x_range), color="#ff6b6b", linewidth=2)
        except Exception:
            pass
        ax2.set_title("Histogram + KDE", color="#cccccc", fontsize=10)
        ax2.tick_params(colors="#aaaaaa")
        for sp in ax2.spines.values():
            sp.set_edgecolor("#333333")

        plt.tight_layout()
        safe = re.sub(r"[^\w]", "_", col_name)
        path = os.path.join(self.boxplot_dir, f"{safe}_boxplot.png")
        plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0f0f0f")
        plt.close(fig)
        return path

    # ── Per-column auditors ──────────────────────────────────────────────────

    def _audit_categorical(self, col_name, series):
        total      = len(series)
        missing    = int(series.isna().sum())
        miss_pct   = round(missing / total * 100, 2)
        vc         = series.value_counts(dropna=True)
        n_distinct = int(vc.shape[0])
        dom_pct    = float(vc.values[0] / vc.values.sum()) if len(vc) > 0 else 0
        distorted  = dom_pct > DISTORTED_CATEGORY_THRESHOLD
        str_vals   = [str(v) for v in vc.index.tolist()]
        split      = self._detect_split(series)
        semantic   = self._detect_semantic(series)
        sample     = (series.dropna()
                            .sample(min(RANDOM_SAMPLE_SIZE, series.dropna().shape[0]), random_state=42)
                            .tolist())
        fuzzy = self._fuzzy_candidates(str_vals)
        embed = self._embed_candidates(str_vals)

        original_name = col_name
        for change in self.report.get("column_header_audit", {}).get("changes", []):
            if change["normalized"] == col_name:
                    original_name = change["original"]
                    break

        recs = []
        if miss_pct >= HIGH_MISSING_THRESHOLD * 100:
            recs.append(f"DROP COLUMN - {miss_pct}% missing exceeds threshold.")
        elif missing > 0:
            recs.append(f"Impute or flag {missing} missing values.")
        if distorted:
            recs.append(f"DISTORTED - '{vc.index[0]}' = {dom_pct*100:.1f}%. Consider collapsing minority classes.")
        if fuzzy:
            t = fuzzy[0]
            recs.append(f"MERGE CANDIDATES - '{t['value_a']}' <-> '{t['value_b']}' ({t['similarity_pct']}% similar).")
        if split["should_split"]:
            recs.append(f"SPLIT CANDIDATE - delimiter '{split['delimiter']}' in {split['hit_rate_pct']}% of values.")
        if semantic:
            recs.append(f"SEMANTIC PATTERN '{semantic}' detected. Consider dedicated column.")
        if not recs:
            recs.append("Column appears clean.")

        return {
            "original_name":              original_name,
            "type":                       "CATEGORICAL",
            "quality_score":              self._quality_score(miss_pct, distorted, False, split["should_split"]),
            "index":                      list(self.df.columns).index(col_name), 
            "total_rows":                 total,
            "missing_count":              missing,
            "missing_pct":                miss_pct,
            "n_distinct":                 n_distinct,
            "all_distinct_values":        vc.index.tolist()[:MAX_CATEGORIES_TO_SHOW],
            "distribution":               [{"value": str(k), "count": int(v),
                                            "pct": round(v / vc.sum() * 100, 2)}
                                           for k, v in vc.items()][:MAX_CATEGORIES_TO_SHOW],
            "dominant_category_pct":      round(dom_pct * 100, 2),
            "is_distorted":               distorted,
            "random_sample":              [str(s) for s in sample],
            "split_analysis":             split,
            "semantic_pattern":           semantic,
            "fuzzy_merge_candidates":     fuzzy,
            "embedding_merge_candidates": embed,
            "recommendations":            recs,
        }

    def _audit_numerical(self, col_name, series):
        total    = len(series)
        missing  = int(series.isna().sum())
        miss_pct = round(missing / total * 100, 2)
        clean    = series.dropna()
        outliers = self._detect_outliers(clean)
        desc     = self._descriptive_stats(clean) if len(clean) > 0 else {}
        split    = self._detect_split(series.astype(str))
        semantic = self._detect_semantic(series.astype(str))
        sample   = ([round(float(s), 4) for s in
                    clean.sample(min(RANDOM_SAMPLE_SIZE, len(clean)), random_state=42).tolist()]
                    if len(clean) > 0 else [])

        # Find original name from the rename changes
        original_name = col_name
        for change in self.report.get("column_header_audit", {}).get("changes", []):
            if change["normalized"] == col_name:
                original_name = change["original"]
                break

        boxplot_path = None
        if len(clean) > 0:
            try:
                boxplot_path = self._make_boxplot(clean, col_name)
            except Exception as e:
                boxplot_path = f"ERROR: {e}"

        recs = []
        if miss_pct >= HIGH_MISSING_THRESHOLD * 100:
            recs.append(f"DROP COLUMN - {miss_pct}% missing exceeds threshold.")
        elif missing > 0:
            recs.append(f"Impute {missing} missing values (median/mean/model).")
        if outliers["outlier_count"] > 0:
            recs.append(f"OUTLIERS - {outliers['outlier_count']} values ({outliers['outlier_pct']}%) "
                        f"via {outliers['method']}. Investigate before modelling.")
        if abs(desc.get("skewness", 0)) > 1 and len(clean) > 0:
            recs.append(f"SKEWED (skew={desc['skewness']}). Consider log/sqrt/Box-Cox transform.")
        if split["should_split"]:
            recs.append(f"SPLIT CANDIDATE - delimiter '{split['delimiter']}' detected.")
        if not recs:
            recs.append("Column appears clean.")

        return {
            "original_name":        original_name,  # ADD THIS
            "type":                 "NUMERICAL",
            "quality_score":        self._quality_score(miss_pct, False,
                                                        outliers["outlier_count"] > 0,
                                                        split["should_split"]),
            "index":                list(self.df.columns).index(col_name),  # ADD INDEX
            "total_rows":           total,
            "missing_count":        missing,
            "missing_pct":          miss_pct,
            "descriptive_stats":    desc,
            "outlier_analysis":     outliers,
            "random_sample":        sample,
            "split_analysis":       split,
            "semantic_pattern":     semantic,
            "boxplot_path":         boxplot_path,
            "recommendations":      recs,
        }
    # ════════════════════════════════════════════════════════════════════════
    # MARKDOWN RENDERER
    # ════════════════════════════════════════════════════════════════════════

    def _render_markdown(self):
        r, lines = self.report, []
        meta = r["meta"]
        lines += [
            "# Dataset Audit Report",
            f"Generated: {meta['generated_at']}  |  "
            f"Rows: {meta['total_rows']:,}  |  Columns: {meta['total_columns']}",
            f"Source: {meta['source_file']}", "",
        ]

        ha = r["column_header_audit"]
        lines += ["## 1. Column Header Audit", ha["recommendation"]]
        if ha["changes"]:
            lines += ["", "| Original | Normalised |", "|---|---|"]
            for ch in ha["changes"]:
                lines.append(f"| {ch['original']} | {ch['normalized']} |")
        lines.append("")

        da = r["duplicate_audit"]
        lines += ["## 2. Duplicate Row Audit",
                f"Duplicate rows: {da['total_duplicate_rows']} ({da['percentage']}%)",
                da["recommendation"], ""]

        s = r["summary"]
        lines += [
            "## 3. Summary",
            f"- Overall quality score: {s['overall_quality_score']} / 100",
            f"- Columns with missing: {s['columns_with_missing']}",
            f"- Drop candidates: {', '.join(s['drop_candidates']) or 'None'}",
            f"- Split candidates: {', '.join(s['split_candidates']) or 'None'}",
            f"- Categorical: {s['categorical_columns']}  |  Numerical: {s['numerical_columns']}", "",
        ]

        lines.append("## 4. Column-Level Audit")
        
        # FIX: Iterate over dictionary items instead of list
        for col_name, col in r["columns"].items():  # CHANGED: .items() instead of direct iteration
            badge = "[GOOD]" if col["quality_score"] >= 80 else ("[WARN]" if col["quality_score"] >= 50 else "[BAD]")
            lines += [
                f"### {badge} {col_name} - {col['type']} | Score: {col['quality_score']}/100",  # CHANGED: use col_name
                f"- Missing: {col['missing_count']} ({col['missing_pct']}%)",
                f"- Random sample: {col['random_sample']}",
            ]
            if col["type"] == "CATEGORICAL":
                lines += [
                    f"- Distinct values: {col['n_distinct']}",
                    f"- Dominant category: {col['dominant_category_pct']}%"
                    + (" [DISTORTED]" if col["is_distorted"] else ""),
                ]
                if col.get("fuzzy_merge_candidates"):
                    t = col["fuzzy_merge_candidates"][0]
                    lines.append(f"- Merge candidate: '{t['value_a']}' <-> '{t['value_b']}' ({t['similarity_pct']}%)")
                lines += ["- Top 10 distribution:", "  | Value | Count | % |", "  |---|---|---|"]
                for row in col["distribution"][:10]:
                    lines.append(f"  | {row['value']} | {row['count']} | {row['pct']}% |")
            else:
                ds, oa = col.get("descriptive_stats", {}), col.get("outlier_analysis", {})
                if ds:
                    lines += [
                        f"- Mean: {ds.get('mean')}  |  Median: {ds.get('median')}  |  Std: {ds.get('std')}",
                        f"- Range: [{ds.get('min')} -> {ds.get('max')}]  |  Skewness: {ds.get('skewness')}",
                    ]
                if oa:
                    lines.append(f"- Outliers ({oa.get('method')}): {oa.get('outlier_count')} ({oa.get('outlier_pct')}%)")
                if col.get("boxplot_path"):
                    lines.append(f"- Boxplot: {col['boxplot_path']}")

            if col.get("split_analysis", {}).get("should_split"):
                sp = col["split_analysis"]
                lines.append(f"- Split on '{sp['delimiter']}' ({sp['hit_rate_pct']}% hit rate)")
            if col.get("semantic_pattern"):
                lines.append(f"- Semantic pattern: {col['semantic_pattern']}")
            lines.append("Recommendations:")
            for rec in col["recommendations"]:
                lines.append(f"  - {rec}")
            lines.append("")

        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# QUICK START — just run:  python dataset_audit_report.py
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    auditor = DatasetAuditor(
        input_path = "C:/Users/tvlan/Downloads/heasrtatatck/heart_attack_prediction_dataset.csv",
        output_dir = "audit_output",
        fmt        = "both",
    )
    auditor.run()
