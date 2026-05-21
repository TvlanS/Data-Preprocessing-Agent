import pandas as pd
import numpy as np
import json
import yaml
import os
import missingno as msno
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import chi2_contingency, mannwhitneyu
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
class MissingValuePipeline:

    def __init__(self, df, col_des, csv_path, exclude_cols=None):
        """
        Parameters
        ----------
        df            : pd.DataFrame  -- input dataset
        col_des       : dict          -- column descriptions yaml loaded as dict
        csv_path      : str           -- original csv path (used for output naming)
        exclude_cols  : list          -- columns to skip during imputation
        """
        self.df            = df.copy()
        self.col_des       = col_des
        self.csv_path      = csv_path
        self.exclude_cols  = exclude_cols or []
        self.timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.original_name = os.path.splitext(os.path.basename(csv_path))[0]
        self.corr          = {}

        # ── Output folder ─────────────────────────────────────────
        self.output_dir = os.path.join(
            os.path.dirname(csv_path),
            f"impute_run_{self.timestamp}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"\n  Output folder : {self.output_dir}")

    # ── Helper : save JSON ────────────────────────────────────────────────────
    def _save_json(self, data, filename):
        path = os.path.join(self.output_dir, f"{filename}_{self.timestamp}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=4, cls=NumpyEncoder)
        print(f"  Saved : {path}")
        return path

    # ── Step 1 : Matrix Plot ──────────────────────────────────────────────────
    def plot_matrix(self, stage="before"):
        print(f"\n{'='*60}")
        print(f"  Missing Value Matrix -- {stage.upper()}")
        print(f"{'='*60}")

        fig, ax = plt.subplots(figsize=(12, 6))
        msno.matrix(self.df, ax=ax, sparkline=False)
        ax.set_title(f"Missing Value Matrix ({stage.upper()}) -- {self.timestamp}", fontsize=13)
        plt.tight_layout()

        plot_path = os.path.join(self.output_dir, f"matrix_{stage}_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=150)
        plt.show()
        print(f"  Saved : {plot_path}")

    # ── Step 2 : Missing Summary -- all columns ───────────────────────────────
    def missing_summary_all(self):
        print(f"\n{'='*60}")
        print(f"  Missing Value Summary -- All Columns")
        print(f"{'='*60}")

        total   = len(self.df)
        summary = []

        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            missing_pct   = round((missing_count / total) * 100, 2)

            print(f"  {col:<45} missing: {missing_count:>4}  ({missing_pct:.2f}%)")
            summary.append({
                "column"       : col,
                "total_rows"   : total,
                "missing_count": int(missing_count),
                "missing_pct"  : missing_pct,
                "excluded"     : col in self.exclude_cols
            })

        self._save_json(summary, "missing_summary")
        return summary

    # ── Step 3 : MNAR Analysis ────────────────────────────────────────────────
    def _profile_missing_rows(self, target_col):
        missing_mask = self.df[target_col].isnull()
        missing_rows = self.df[missing_mask]
        present_rows = self.df[~missing_mask]
        profile      = []

        print(f"\n{'='*60}")
        print(f"  Profile of rows where '{target_col}' is MISSING")
        print(f"  Missing count: {missing_mask.sum()} / {len(self.df)} rows")
        print(f"{'='*60}")

        for col in self.df.columns:
            if col == target_col:
                continue
            if self.df[col].isnull().all():
                continue

            is_numeric     = pd.api.types.is_numeric_dtype(self.df[col])
            is_categorical = (
                pd.api.types.is_string_dtype(self.df[col])           or
                pd.api.types.is_object_dtype(self.df[col])           or
                isinstance(self.df[col].dtype, pd.CategoricalDtype)  or
                (not is_numeric)                                      or
                self.df[col].nunique() <= 10
            )

            if is_categorical:
                miss_dist    = missing_rows[col].value_counts(normalize=True)
                present_dist = present_rows[col].value_counts(normalize=True)

                dominant_val_missing = miss_dist.idxmax()    if not miss_dist.empty    else None
                dominant_pct_missing = miss_dist.max()       if not miss_dist.empty    else 0
                dominant_val_present = present_dist.idxmax() if not present_dist.empty else None
                dominant_pct_present = present_dist.max()    if not present_dist.empty else 0

                ct = pd.crosstab(missing_mask, self.df[col])
                try:
                    _, p_val, _, _ = chi2_contingency(ct)
                except Exception:
                    p_val = 1.0

                profile.append({
                    "column"               : col,
                    "type"                 : "categorical",
                    "dominant_val_missing" : dominant_val_missing,
                    "dominant_pct_missing" : round(dominant_pct_missing * 100, 2),
                    "dominant_val_present" : dominant_val_present,
                    "dominant_pct_present" : round(dominant_pct_present * 100, 2),
                    "mean_when_missing"    : None,
                    "median_when_missing"  : None,
                    "mean_when_present"    : None,
                    "direction"            : None,
                    "p_value"              : round(p_val, 4),
                    "significant"          : p_val < 0.05
                })

                print(f"\n  [{col}] -- categorical")
                print(f"    Most common when '{target_col}' missing : "
                      f"'{dominant_val_missing}' ({dominant_pct_missing*100:.1f}%)")
                print(f"    Most common when '{target_col}' present : "
                      f"'{dominant_val_present}' ({dominant_pct_present*100:.1f}%)")
                print(f"    Chi-square p = {p_val:.4f}  "
                      f"{'significant' if p_val < 0.05 else 'no signal'}")

            else:
                miss_vals    = pd.to_numeric(missing_rows[col], errors="coerce").dropna()
                present_vals = pd.to_numeric(present_rows[col], errors="coerce").dropna()

                if len(miss_vals) < 3 or len(present_vals) < 3:
                    continue

                miss_mean    = miss_vals.mean()
                present_mean = present_vals.mean()
                miss_median  = miss_vals.median()
                direction    = "higher" if miss_mean > present_mean else "lower"

                try:
                    _, p_val = mannwhitneyu(miss_vals, present_vals, alternative="two-sided")
                except Exception:
                    p_val = 1.0

                profile.append({
                    "column"               : col,
                    "type"                 : "numerical",
                    "dominant_val_missing" : None,
                    "dominant_pct_missing" : None,
                    "dominant_val_present" : None,
                    "dominant_pct_present" : None,
                    "mean_when_missing"    : round(miss_mean, 2),
                    "median_when_missing"  : round(miss_median, 2),
                    "mean_when_present"    : round(present_mean, 2),
                    "direction"            : direction,
                    "p_value"              : round(p_val, 4),
                    "significant"          : p_val < 0.05
                })

                print(f"\n  [{col}] -- numerical")
                print(f"    Mean when '{target_col}' missing : {miss_mean:.2f}  "
                      f"(median: {miss_median:.2f})")
                print(f"    Mean when '{target_col}' present : {present_mean:.2f}")
                print(f"    Direction : missing rows have {direction} {col}")
                print(f"    Mann-Whitney p = {p_val:.4f}  "
                      f"{'significant' if p_val < 0.05 else 'no signal'}")

        return pd.DataFrame(profile)

    def run_mnar_analysis(self, min_signals=1):
        print(f"\n{'='*60}")
        print(f"  MNAR Analysis")
        print(f"{'='*60}")

        target_cols = [
            c for c in self.df.columns
            if self.df[c].isnull().any() and c not in self.exclude_cols
        ]

        output = {}

        for col in target_cols:
            profile_df          = self._profile_missing_rows(col)
            if profile_df.empty:
                continue

            significant_drivers = profile_df["significant"].sum()
            total_tested        = len(profile_df)
            signal_ratio        = round(significant_drivers / total_tested, 3) if total_tested > 0 else 0

            if significant_drivers < min_signals:
                continue

            top_signals = profile_df[profile_df["significant"]].sort_values("p_value")
            drivers     = []

            for _, row in top_signals.iterrows():
                if row["type"] == "categorical":
                    drivers.append({
                        "column"               : row["column"],
                        "dominant_val_missing" : row["dominant_val_missing"],
                        "dominant_pct_missing" : row["dominant_pct_missing"],
                        "dominant_val_present" : row["dominant_val_present"],
                        "dominant_pct_present" : row["dominant_pct_present"],
                        "p_value"              : row["p_value"]
                    })
                else:
                    drivers.append({
                        "column"             : row["column"],
                        "mean_when_missing"  : row["mean_when_missing"],
                        "median_when_missing": row["median_when_missing"],
                        "mean_when_present"  : row["mean_when_present"],
                        "direction"          : row["direction"],
                        "p_value"            : row["p_value"]
                    })

            output[col] = {
                "significant_drivers": int(significant_drivers),
                "total_tested"       : int(total_tested),
                "signal_ratio"       : signal_ratio,
                "drivers"            : drivers
            }

        self.corr = output
        self._save_json(output, "mnar_analysis")
        return output

    # ── Step 4 : Imputation Handler ───────────────────────────────────────────
    def _missing_value_handler(self, target_col):
        is_numeric   = pd.api.types.is_numeric_dtype(self.df[target_col])
        col          = self.df[target_col]
        missing_mask = self.df[target_col].isna()
        missing_rows = self.df[missing_mask].copy()
        present_rows = self.df[~missing_mask].copy()
        total        = len(missing_rows)

        if total == 0:
            print(f"\n  No missing values found for '{target_col}'.")
            return {}

        # ── Column stats ──────────────────────────────────────────
        if is_numeric:
            col_numeric = pd.to_numeric(col, errors="coerce")
            mean        = col_numeric.mean()
            median      = col_numeric.median()
            Q1          = col_numeric.quantile(0.25)
            Q3          = col_numeric.quantile(0.75)
            IQR         = Q3 - Q1
            lower       = Q1 - 1.5 * IQR
            upper       = Q3 + 1.5 * IQR

            missing_bounds = {
                "column"        : target_col,
                "type"          : "numerical",
                "mean"          : round(mean, 4),
                "median"        : round(median, 4),
                "Q1"            : round(Q1, 4),
                "Q3"            : round(Q3, 4),
                "IQR"           : round(IQR, 4),
                "lower_1_5_IQR" : round(lower, 4),
                "upper_1_5_IQR" : round(upper, 4),
                "total_missing" : total,
            }
        else:
            value_counts     = present_rows[target_col].value_counts()
            value_counts_pct = present_rows[target_col].value_counts(normalize=True).round(4)
            dominant_val     = value_counts.idxmax() if not value_counts.empty else None
            dominant_pct     = value_counts_pct.max() if not value_counts_pct.empty else 0

            missing_bounds = {
                "column"           : target_col,
                "type"             : "categorical",
                "unique_values"    : col.dropna().unique().tolist(),
                "value_counts"     : value_counts.to_dict(),
                "value_counts_pct" : value_counts_pct.to_dict(),
                "dominant_value"   : dominant_val,
                "dominant_pct"     : round(dominant_pct * 100, 2),
                "total_missing"    : total,
            }

        missing_records = []
        all_indices     = self.df.index.tolist()

        for i, (idx, row) in enumerate(missing_rows.iterrows(), start=1):

            # ── 5 closest neighbours ──────────────────────────────
            idx_pos    = all_indices.index(idx)
            neighbours = []
            offset     = 1

            while len(neighbours) < 5:
                for direction in [-1, 1]:
                    neighbour_pos = idx_pos + (direction * offset)
                    if 0 <= neighbour_pos < len(all_indices):
                        neighbour_idx = all_indices[neighbour_pos]
                        if not pd.isna(self.df.at[neighbour_idx, target_col]):
                            neighbours.append(neighbour_idx)
                    if len(neighbours) == 5:
                        break
                offset += 1
                if offset > len(all_indices):
                    break

            neighbours         = neighbours[:5]
            closest_neighbours = [self.df.at[n_idx, target_col] for n_idx in neighbours]

            records = {
                "index"             : i,
                "total"             : total,
                "row_index"         : idx,
                "values"            : {
                    target_col: None,
                    **{
                        col_name: (None if pd.isna(row[col_name]) else row[col_name])
                        for col_name in self.df.columns
                        if col_name != target_col
                    }
                },
                "closest_neighbours": closest_neighbours
            }

            print(records)

            try:
                correlated = self.corr[target_col]
            except Exception:
                correlated = "null"

            # ── LLM Call ──────────────────────────────────────────
            prompt          = Config()
            cleaning_prompt = prompt.missing_prompt
            data_sum        = self.col_des["Summary"]
            col_sum         = self.col_des["Column"][target_col]

            cleaning_prompt = cleaning_prompt.replace("{dataset summary}", data_sum)
            cleaning_prompt = cleaning_prompt.replace("{column description}", col_sum)
            cleaning_prompt = cleaning_prompt.replace("{column}", target_col)

            slm_model     = llm(cleaning_prompt, f"{missing_bounds}\n {records} \n Correlated Pair:\n{correlated}")
            cleaned_value = slm_model.llm_call()
            print(cleaned_value)
            cleaned_value            = json.loads(cleaned_value)
            records["cleaned_value"] = cleaned_value["action"]
            records["reason"]        = cleaned_value["reason"]

            missing_records.append(records)

        return {
            "missing_bounds" : missing_bounds,
            "missing_records": missing_records
        }

    # ── Step 5 : Apply Cleaned Values ─────────────────────────────────────────
    def _apply_cleaned_values(self, output):
        target_col = output["missing_bounds"]["column"]
        records    = output["missing_records"]

        print(f"\n  Applying cleaned values for '{target_col}'")
        print(f"{'='*60}")

        for record in records:
            row_index     = record["row_index"]
            cleaned_value = record["cleaned_value"]

            if pd.api.types.is_numeric_dtype(self.df[target_col]):
                try:
                    cleaned_value = float(cleaned_value)
                except (ValueError, TypeError):
                    print(f"  Could not convert '{cleaned_value}' for row {row_index}, skipping.")
                    continue

            self.df.at[row_index, target_col] = cleaned_value
            print(f"  Row {row_index:>5}  ->  {target_col} = {cleaned_value}")

        print(f"{'='*60}")
        print(f"  {len(records)} values updated.\n")

    # ── Run Full Pipeline ─────────────────────────────────────────────────────
    def run(self):
        print(f"\n{'='*60}")
        print(f"  Missing Value Pipeline -- {self.timestamp}")
        print(f"{'='*60}")

        # ── Step 1 : Matrix before ────────────────────────────────
        self.plot_matrix(stage="before")

        # ── Step 2 : Missing summary ──────────────────────────────
        self.missing_summary_all()

        # ── Step 3 : MNAR analysis (generates self.corr) ──────────
        self.run_mnar_analysis()

        # ── Step 4 & 5 : Impute + Apply per column ────────────────
        missing_cols = [
            c for c in self.df.columns
            if self.df[c].isnull().any() and c not in self.exclude_cols
        ]

        for col in missing_cols:
            print(f"\n{'='*60}")
            print(f"  Imputing : {col}")
            print(f"{'='*60}")

            output = self._missing_value_handler(col)

            if not output:
                continue

            self._save_json(output, f"impute_{col}")
            self._apply_cleaned_values(output)

        # ── Step 6 : Matrix after ─────────────────────────────────
        self.plot_matrix(stage="after")

        # ── Step 7 : Final missing summary ────────────────────────
        print(f"\n{'='*60}")
        print(f"  Final Missing Value Summary")
        print(f"{'='*60}")
        self.missing_summary_all()

        # ── Step 8 : Save imputed dataset ─────────────────────────
        output_csv_name = f"impute_{self.original_name}_{self.timestamp}.csv"
        output_csv_path = os.path.join(self.output_dir, output_csv_name)
        self.df.to_csv(output_csv_path, index=False)
        print(f"\n  Imputed dataset saved : {output_csv_path}")

        return self.df


# ── Run ───────────────────────────────────────────────────────────────────────
path_yaml = rf"C:\Users\tvlan\Documents\Data Mining\1.0 Assignment\1.0 Python\reference\tb_col.yaml"
path_csv  = rf"C:\Users\tvlan\Documents\Data Mining\1.0 Assignment\1.0 Python\Data\outlier_run_20260426_151523\outlier_tb_syth_data_basic_v4_20260426_151523.csv"

with open(path_yaml, "r", encoding="utf-8") as f:
    col_des = yaml.safe_load(f)

df = pd.read_csv(path_csv)

pipeline = MissingValuePipeline(
    df           = df,
    col_des      = col_des,
    csv_path     = path_csv,
    exclude_cols = ["First_Symptoms"]   # -- list any columns to skip here
)

df_imputed = pipeline.run()