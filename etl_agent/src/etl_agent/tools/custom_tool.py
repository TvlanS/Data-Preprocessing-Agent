from typing import Type

from pydantic import BaseModel, Field
from pyprojroot import here
import sys

sys.path.insert(0,str(here()))

from crewai.tools import BaseTool ,tool
from toolbox.normalise_cleaning_tool import setup_run

from scipy.stats import pearsonr
import pandas as pd
import random          
import json


@tool("ask_user")
def ask_user(question: str) -> str:
            """Ask the human user a question and return their answer."""
            return input(f"\n[Agent asks] {question}\n> ")

class normalise_cleaning_tool_Input(BaseModel):

    dataset_path : str = Field (description = "The path of the original dataset")
    row_threshold : int = Field(default= 70 , description= "Drop row with more than this % of empty cells (e.g. ``70.0``)")
    col_threshold : int = Field(default= 70 , description= "Drop columns with more than this % of empty cells (e.g. ``70.0``)")
    minimal_profilling : bool = Field(default = False , description = """ Pass ``True`` to run a lighter-weight profiling scan (skips
                                                                           correlation analysis). Recommended for datasets > 100k rows.""") 
class pearson_tool_Input(BaseModel):

    dataset_path: str = Field(
        description="The path to the dataset file to load and perform Pearson correlation analysis on."
    )

    target_feature: str = Field(
        description="The name of the target column in the dataset against which all other feature correlations will be computed (e.g. 'Treatment_duration')."
    )

class describe_dataset_tool_Input(BaseModel):
    dataset_path: str = Field(
        description="Path to the CSV dataset to profile and describe."
    )


class normalise_cleaning_tool(BaseTool):
    name: str = "normalise cleaning tool"
    description: str = "First step of dataset cleaning which includes column normalisation, empty row and column removal"
    args_schema: Type[BaseModel] = normalise_cleaning_tool_Input

    def _run(self,dataset_path : str,row_threshold : int,col_threshold : int,minimal_profilling : bool) -> str:
        from toolbox.normalise_cleaning_tool import DataCleaningTool, _load_dataset

        run_dir, config = setup_run(
            dataset_path,
            row_threshold=float(row_threshold),
            col_threshold=float(col_threshold),
            minimal_profiling=minimal_profilling,
        )
        df = _load_dataset(dataset_path)
        tool = DataCleaningTool(run_dir, float(row_threshold), float(col_threshold), minimal_profilling)
        df_clean = tool.run(df)
        return f"Cleaning complete. {len(df_clean)} rows written to {run_dir}"

class describe_dataset_tool(BaseTool):
    name: str = "Dataset Description Tool"
    description: str = (
        "Profiles a dataset and returns a JSON summary including total dataset size "
        "and each column's data type, number of unique values, null count, and 3 random "
        "sample values. Use this BEFORE running Pearson correlation to decide which "
        "columns to include, encode, or skip."
    )
    args_schema: Type[BaseModel] = describe_dataset_tool_Input

    def _run(self, dataset_path: str) -> str:

        df = pd.read_csv(dataset_path)

        description = {
            "dataset_info": {
                "total_rows": int(len(df)),
                "total_columns": int(len(df.columns)),
                "total_cells": int(df.size),
                "total_null_cells": int(df.isnull().sum().sum()),
                "null_pct_overall": round(df.isnull().mean().mean() * 100, 2)
            },
            "columns": {}
        }

        for col in df.columns:
            sample_values = df[col].dropna().tolist()
            random_samples = random.sample(sample_values, min(3, len(sample_values)))  # ✅ full line

            description["columns"][col] = {
                "dtype": str(df[col].dtype),
                "num_unique": int(df[col].nunique()),
                "null_count": int(df[col].isnull().sum()),
                "null_pct": round(df[col].isnull().mean() * 100, 2),
                "random_samples": [str(s) for s in random_samples]
            }
        output = json.dumps(description, indent=2)
        print(output)
        return output

class pearson_tool(BaseTool):
    name: str = "Pearson Correlation Tool"
    description: str = "Performs correlation analysis to identify the correlation between target and other features"
    args_schema: Type[BaseModel] = pearson_tool_Input

    def _run(self,dataset_path : str, target_feature : str) -> str:
        pearson_real = {}

        df_ori = pd.read_csv(dataset_path)
        df_ori =pd.DataFrame(df_ori)

        for col in df_ori.columns:
            if col != target_feature:
                corr,p_value = pearsonr(df_ori[col],df_ori[target_feature])
                pearson_real[col] = {"Pearson Correlation Real":corr,"p_value Real":round(p_value,4)}

        pearson_table_real = pd.DataFrame.from_dict(pearson_real).T
        pearson_table_real = pearson_table_real.reindex(pearson_table_real["Pearson Correlation Real"].abs().sort_values(ascending=False).index)

        return pearson_table_real.to_string()



if __name__ == "__main__":
    pear = describe_dataset_tool()
    path = r"C:\Users\tvlan\Documents\1.0 Python\5.0 Automated Cleaning Agent\data\animation_movies_enriched_1878_2029.csv"
    pear._run(path)
  
      