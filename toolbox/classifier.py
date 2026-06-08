from pyprojroot import here
import sys
import pandas as pd
import yaml
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

sys.path.append(str(here()))

from Utils.LLM_load import llm
from Utils.config_setup import Config


class SymptomClassifier:
    """
    Classifies raw symptom text in a CSV using an LLM, saves the enriched
    data as a new CSV copy, and produces a frequency bar chart.

    Parameters
    ----------
    csv_path        : path to the source CSV file
    yaml_path       : path to the column-description YAML
    output_csv_path : path for the new output CSV (optional).
                      Defaults to <original_name>_classified.csv
                      in the same folder as the source file.
    column          : name of the raw symptom column to classify
    """

    def __init__(
        self,
        csv_path: str,
        yaml_path: str,
        output_csv_path: str | None = None,
        column: str = "First_Symptoms",
    ):
        self.csv_path   = csv_path
        self.yaml_path  = yaml_path
        self.column     = column
        self.new_column = f"{column}_Classified"

        # Auto-generate output path if not supplied
        if output_csv_path is None:
            p = Path(csv_path)
            output_csv_path = str(p.with_stem(p.stem + "_classified"))
        self.output_csv_path = output_csv_path

        self.df: pd.DataFrame | None = None
        self._col_des: dict          = {}

    # ------------------------------------------------------------------
    # 1.  Load
    # ------------------------------------------------------------------
    def load(self) -> "SymptomClassifier":
        """Load the YAML metadata and the source CSV."""
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            self._col_des = yaml.safe_load(f)

        self.df = pd.read_csv(self.csv_path)
        print(f"[load] Loaded {len(self.df):,} rows from {self.csv_path}")
        return self

    # ------------------------------------------------------------------
    # 2.  Inspect raw distinct values BEFORE classification
    # ------------------------------------------------------------------
    def show_distinct_raw(self) -> pd.Series:
        """
        Print (and return) the distinct atomic values in the raw column.
        Values are split on commas so multi-symptom cells are expanded.
        """
        self._check_loaded()
        distinct = (
            self.df[self.column]
            .dropna()
            .str.split(",")
            .explode()
            .str.strip()
            .unique()
        )
        distinct_sorted = sorted(distinct)

        print(f"\n{'='*55}")
        print(f"  Distinct raw values in '{self.column}' "
              f"({len(distinct_sorted)} total)")
        print(f"{'='*55}")
        for v in distinct_sorted:
            print(f"    {v}")

        return pd.Series(distinct_sorted, name=self.column)

    # ------------------------------------------------------------------
    # 3.  Classify
    # ------------------------------------------------------------------
    def classify(self) -> "SymptomClassifier":
        """Run LLM classification row-by-row and store results in a new column."""
        self._check_loaded()
        self.df[self.new_column] = None

        print(f"\n[classify] Classifying {len(self.df):,} rows ...")
        for i in range(len(self.df)):
            prompt       = Config()
            clean_prompt = prompt.classification_prompt
            raw_value    = self.df[self.column].iloc[i]
            model        = llm(clean_prompt, str(raw_value))
            response     = model.llm_call()

            classified                     = json.loads(response)["action"]
            self.df.at[i, self.new_column] = classified

            # Progress feedback every 50 rows
            if (i + 1) % 50 == 0 or i == 0:
                print(f"    [{i+1}/{len(self.df)}] {raw_value!r} -> {classified!r}")

        print(f"[classify] Done. Column '{self.new_column}' added.")
        return self

    # ------------------------------------------------------------------
    # 4a. Distinct counts for the NEW column
    # ------------------------------------------------------------------
    def distinct_counts_classified(self) -> pd.DataFrame:
        """Return (and print) a frequency table of the newly classified column."""
        self._check_classified()
        counts = (
            self.df[self.new_column]
            .value_counts()
            .rename_axis(self.new_column)
            .reset_index(name="count")
        )

        print(f"\n{'='*55}")
        print(f"  Distinct counts for '{self.new_column}'")
        print(f"{'='*55}")
        print(counts.to_string(index=False))
        return counts

    # ------------------------------------------------------------------
    # 4b. Frequency bar chart for the NEW column
    # ------------------------------------------------------------------
    def plot_frequency(
        self,
        save_path: str | None = None,
        top_n: int | None = None,
    ) -> None:
        """
        Draw a horizontal bar chart of classified-label frequencies.

        Parameters
        ----------
        save_path : if given, save the figure to this path (PNG / SVG ...)
        top_n     : show only the top-N categories (all by default)
        """
        self._check_classified()
        counts = (
            self.df[self.new_column]
            .value_counts()
            .sort_values()          # ascending -> longest bar at top
        )
        if top_n:
            counts = counts.tail(top_n)

        fig, ax = plt.subplots(figsize=(10, max(4, len(counts) * 0.5)))
        bars = ax.barh(counts.index, counts.values, color="#4C72B0", edgecolor="white")

        # Value labels on each bar
        for bar, val in zip(bars, counts.values):
            ax.text(
                bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", ha="left", fontsize=9,
            )

        ax.set_xlabel("Count", fontsize=11)
        ax.set_title(
            f"Frequency of '{self.new_column}'"
            + (f"  (Top {top_n})" if top_n else ""),
            fontsize=13, fontweight="bold",
        )
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"[plot] Chart saved to {save_path}")
        plt.show()

    # ------------------------------------------------------------------
    # 5.  Save enriched data as a NEW CSV copy
    # ------------------------------------------------------------------
    def save_to_csv(self) -> "SymptomClassifier":
        """
        Write the full enriched DataFrame (including the new classified column)
        to a new CSV file. The original source file is never modified.
        """
        self._check_classified()
        self.df.to_csv(self.output_csv_path, index=False, encoding="utf-8")
        print(f"\n[save] {len(self.df):,} rows saved to -> {self.output_csv_path}")
        return self

    # ------------------------------------------------------------------
    # 6.  Full pipeline convenience method
    # ------------------------------------------------------------------
    def run(self, chart_save_path: str | None = None) -> pd.DataFrame:
        """
        Execute the full pipeline in order:
          load -> show raw distinct -> classify
          -> distinct counts -> frequency chart -> save new CSV

        Returns the enriched DataFrame.
        """
        self.load()
        self.show_distinct_raw()           # step 2 - raw distinct BEFORE classification
        self.classify()                    # step 3
        self.distinct_counts_classified()  # step 4a - distinct counts for new column
        self.plot_frequency(save_path=chart_save_path)  # step 4b - frequency chart
        self.save_to_csv()                 # step 5 - new CSV copy
        return self.df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _check_loaded(self):
        if self.df is None:
            raise RuntimeError("Call .load() first.")

    def _check_classified(self):
        self._check_loaded()
        if self.new_column not in self.df.columns:
            raise RuntimeError("Call .classify() before this step.")


# ======================================================================
# Usage example
# ======================================================================
if __name__ == "__main__":
    CSV_PATH  = r"C:\Users\tvlan\Documents\Data Mining\1.0 Assignment\1.0 Python\Data\outlier_run_20260426_151523\impute_run_20260426_175754\impute_outlier_tb_syth_data_basic_v4_20260426_151523_20260426_175754.csv"
    YAML_PATH = r"C:\Users\tvlan\Documents\1.0 Python\5.0 Automated Cleaning Agent\reference\tb_col.yaml"

    classifier = SymptomClassifier(
        csv_path  = CSV_PATH,
        yaml_path = YAML_PATH,
        # output_csv_path defaults to: tb_syth_data_basic_v5_classified.csv
        column    = "First_Symptoms",
    )

    # --- Run the full pipeline in one call ---
    enriched_df = classifier.run(
        chart_save_path = "frequency_chart.png",  # None -> display only, don't save
    )

    # --- Or call each step individually ---
    # classifier.load()
    # classifier.show_distinct_raw()
    # classifier.classify()
    # classifier.distinct_counts_classified()
    # classifier.plot_frequency(top_n=20)
    # classifier.save_to_csv()
