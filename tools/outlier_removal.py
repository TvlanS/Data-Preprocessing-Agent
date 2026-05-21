import pandas as pd
import numpy as np
import json
import yaml
import os
import matplotlib.pyplot as plt
from datetime import datetime
from pyprojroot import here
import sys

sys.path.append(str(here()))
from Utils.LLM_load import llm
from Utils.config_setup import Config


# ── Numpy JSON Encoder ────────────────────────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ── Main Pipeline Class ───────────────────────────────────────────────────────
class OutlierRemovalPipeline:

    def __init__(self, df, col_des, csv_path, exclude_cols=None):
        """
        Parameters
        ----------
        df            : pd.DataFrame  -- input dataset
        col_des       : dict          -- column descriptions yaml loaded as dict
        csv_path      : str           -- original csv path (used for output naming)
        exclude_cols  : list          -- numerical columns to skip
        """
        self.df            = df.copy()
        self.col_des       = col_des
        self.csv_path      = csv_path
        self.exclude_cols  = exclude_cols or []
        self.timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.original_name = os.path.splitext(os.path.basename(csv_path))[0]

        # ── Identify numerical columns ─────────────────────────────
        self.num_cols = [
            c for c in df.select_dtypes(include=["float64", "int64"]).columns
            if c not in self.exclude_cols
        ]

        # ── Output folder ──────────────────────────────────────────
        self.output_dir = os.path.join(
            os.path.dirname(csv_path),
            f"outlier_run_{self.timestamp}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"\n  Output folder : {self.output_dir}")
        print(f"  Numerical columns : {self.num_cols}")

    # ── Helper : save JSON ────────────────────────────────────────────────────
    def _save_json(self, data, filename):
        path = os.path.join(self.output_dir, f"{filename}_{self.timestamp}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=4, cls=NumpyEncoder)
        print(f"  Saved : {path}")
        return path

    # ── Step 1 : Boxplots ─────────────────────────────────────────────────────
    def plot_boxplots(self, stage="before"):
        print(f"\n{'='*60}")
        print(f"  Boxplots -- {stage.upper()}")
        print(f"{'='*60}")

        n     = len(self.num_cols)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        axes      = np.array(axes).flatten()

        for i, col in enumerate(self.num_cols):
            data = self.df[col].dropna()
            axes[i].boxplot(data, vert=True, patch_artist=True,
                            boxprops=dict(facecolor="#B5D4F4", color="#185FA5"),
                            medianprops=dict(color="#185FA5", linewidth=2),
                            whiskerprops=dict(color="#185FA5"),
                            capprops=dict(color="#185FA5"),
                            flierprops=dict(marker="o", markerfacecolor="#E24B4A",
                                            markersize=5, linestyle="none"))
            axes[i].set_title(col, fontsize=11)
            axes[i].set_xlabel("")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"Boxplots ({stage.upper()}) -- {self.timestamp}", fontsize=13)
        plt.tight_layout()

        plot_path = os.path.join(self.output_dir, f"boxplots_{stage}_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=150)
        plt.show()
        print(f"  Saved : {plot_path}")

    # ── Step 2 : Outlier Summary ──────────────────────────────────────────────
    def _get_outlier_count(self, col_name):
        col   = self.df[col_name].dropna()
        Q1    = col.quantile(0.25)
        Q3    = col.quantile(0.75)
        IQR   = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return int(((col > upper) | (col < lower)).sum())

    def outlier_summary(self, stage="before"):
        print(f"\n{'='*60}")
        print(f"  Outlier Summary -- {stage.upper()}")
        print(f"{'='*60}")

        summary = []
        total   = len(self.df)

        for col in self.num_cols:
            count = self._get_outlier_count(col)
            pct   = round((count / total) * 100, 2)
            print(f"  {col:<35} outliers: {count:>4}  ({pct:.2f}%)")
            summary.append({
                "column"        : col,
                "stage"         : stage,
                "total_rows"    : total,
                "outlier_count" : count,
                "outlier_pct"   : pct
            })

        self._save_json(summary, f"outlier_summary_{stage}")
        return summary

    # ── Step 3 : Outlier Removal Handler ─────────────────────────────────────
    def _outlier_removal(self, target_col):

        df_num = self.df.select_dtypes(include=["float64", "int64"])

        # ── Pearson top 5 correlated columns ──────────────────────
        correlations = (
            df_num.corr(method="pearson")[target_col]
            .drop(target_col)
            .abs()
            .sort_values(ascending=False)
        )
        top5_cols = correlations.head(5).index.tolist()

        # ── IQR bounds ────────────────────────────────────────────
        col    = df_num[target_col]
        Q1     = col.quantile(0.25)
        Q3     = col.quantile(0.75)
        IQR    = Q3 - Q1
        lower  = Q1 - 1.5 * IQR
        upper  = Q3 + 1.5 * IQR
        mean   = col.mean()
        median = col.median()

        outlier_mask = (col > upper) | (col < lower)
        outlier_rows = self.df[outlier_mask].copy()
        outlier_rows["Outlier Direction"] = np.where(
            self.df.loc[outlier_mask, target_col] > upper,
            "ABOVE upper fence",
            "BELOW lower fence"
        )
        total = len(outlier_rows)

        if total == 0:
            print(f"\n  No outliers found for '{target_col}'.")
            return {}

        outlier_bounds = {
            "column"        : target_col,
            "mean"          : round(mean, 4),
            "median"        : round(median, 4),
            "Q1"            : round(Q1, 4),
            "Q3"            : round(Q3, 4),
            "IQR"           : round(IQR, 4),
            "lower_1_5_IQR" : round(lower, 4),
            "upper_1_5_IQR" : round(upper, 4),
        }

        outlier_records = []

        for i, (idx, row) in enumerate(outlier_rows.iterrows(), start=1):
            direction = row["Outlier Direction"]

            records = {
                "index"     : i,
                "total"     : total,
                "row_index" : idx,
                "direction" : direction,
                "values"    : {
                    target_col: row[target_col],
                    **{col_name: row[col_name] for col_name in top5_cols}
                }
            }

            # ── LLM Decision ──────────────────────────────────────
            prompt          = Config()
            cleaning_prompt = prompt.cleaning_prompt
            data_sum        = self.col_des["Summary"]
            col_sum         = self.col_des["Column"][target_col]

            cleaning_prompt = cleaning_prompt.replace("{dataset summary}", data_sum)
            cleaning_prompt = cleaning_prompt.replace("{column description}", col_sum)
            cleaning_prompt = cleaning_prompt.replace("{column}", target_col)

            slm_model     = llm(cleaning_prompt, f"{outlier_bounds}\n {records}")
            cleaned_value = slm_model.llm_call()
            print(cleaning_prompt)
            print(f"{outlier_bounds}\n {records}")
            cleaned_value            = json.loads(cleaned_value)
            records["cleaned_value"] = cleaned_value["action"]
            records["reason"]        = cleaned_value["reason"]

            outlier_records.append(records)

        return {
            "outlier_bounds" : outlier_bounds,
            "outlier_records": outlier_records
        }

    # ── Step 4 : Apply Decisions ──────────────────────────────────────────────
    def _apply_outlier_decisions(self, output):
        target_col = output["outlier_bounds"]["column"]
        records    = output["outlier_records"]

        rows_to_remove = [
            r["row_index"] for r in records
            if r["cleaned_value"] == "remove"
        ]

        print(f"\n  Applying decisions for '{target_col}'")
        print(f"{'='*60}")
        print(f"  Total outliers   : {len(records)}")
        print(f"  Marked remove    : {len(rows_to_remove)}")
        print(f"  Marked keep      : {len(records) - len(rows_to_remove)}")

        self.df = self.df.drop(index=rows_to_remove).reset_index(drop=True)

        print(f"  Rows remaining   : {len(self.df)}")
        print(f"{'='*60}")

    # ── Step 5 : Final Comparison Summary ────────────────────────────────────
    def _final_comparison(self, before_summary, after_summary):
        print(f"\n{'='*60}")
        print(f"  Final Outlier Comparison")
        print(f"{'='*60}")

        comparison = []

        before_map = {r["column"]: r for r in before_summary}
        after_map  = {r["column"]: r for r in after_summary}

        for col in self.num_cols:
            b          = before_map.get(col, {})
            a          = after_map.get(col, {})
            b_count    = b.get("outlier_count", 0)
            a_count    = a.get("outlier_count", 0)
            removed    = b_count - a_count
            pct_reduce = round((removed / b_count * 100), 2) if b_count > 0 else 0.0

            print(f"  {col:<35} before: {b_count:>4}  after: {a_count:>4}  "
                  f"removed: {removed:>4}  reduced: {pct_reduce:.2f}%")

            comparison.append({
                "column"            : col,
                "outliers_before"   : b_count,
                "outliers_after"    : a_count,
                "outliers_removed"  : removed,
                "pct_reduced"       : pct_reduce
            })

        self._save_json(comparison, "outlier_comparison")
        return comparison

    # ── Run Full Pipeline ─────────────────────────────────────────────────────
    def run(self):
        print(f"\n{'='*60}")
        print(f"  Outlier Removal Pipeline -- {self.timestamp}")
        print(f"{'='*60}")

        # ── Step 1 : Boxplots before ──────────────────────────────
        self.plot_boxplots(stage="before")

        # ── Step 2 : Outlier summary before ───────────────────────
        before_summary = self.outlier_summary(stage="before")

        # ── Step 3 & 4 : Detect + apply per column ────────────────
        for col in self.num_cols:
            print(f"\n{'='*60}")
            print(f"  Processing : {col}")
            print(f"{'='*60}")

            output = self._outlier_removal(col)

            if not output:
                continue

            self._save_json(output, f"outlier_decisions_{col}")
            self._apply_outlier_decisions(output)

        # ── Step 5 : Boxplots after ───────────────────────────────
        self.plot_boxplots(stage="after")

        # ── Step 6 : Outlier summary after ────────────────────────
        after_summary = self.outlier_summary(stage="after")

        # ── Step 7 : Final comparison ─────────────────────────────
        self._final_comparison(before_summary, after_summary)

        # ── Step 8 : Save cleaned dataset ─────────────────────────
        output_csv_name = f"outlier_{self.original_name}_{self.timestamp}.csv"
        output_csv_path = os.path.join(self.output_dir, output_csv_name)
        self.df.to_csv(output_csv_path, index=False)
        print(f"\n  Cleaned dataset saved : {output_csv_path}")

        return self.df


# ── Run ───────────────────────────────────────────────────────────────────────
path_yaml = rf"C:\Users\tvlan\Documents\Data Mining\1.0 Assignment\1.0 Python\reference\tb_col.yaml"
path_csv  = rf"C:\Users\tvlan\Documents\Data Mining\1.0 Assignment\1.0 Python\Data\tb_syth_data_basic_v4.csv"

with open(path_yaml, "r", encoding="utf-8") as f:
    col_des = yaml.safe_load(f)

df = pd.read_csv(path_csv)

pipeline = OutlierRemovalPipeline(
    df           = df,
    col_des      = col_des,
    csv_path     = path_csv,
    exclude_cols = []          # -- add any numerical columns to skip here
)

df_cleaned = pipeline.run()