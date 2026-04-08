"""
tools/builtin_tools.py
----------------------
Pre-built pandas cleaning functions.
Every function must follow this signature:

    func(df: pd.DataFrame, column: str, params: dict) -> pd.DataFrame

Add new built-in ops here and register them in the TOOLS dict at the bottom.
Custom (AI-generated, human-approved) tools are stored in custom_tools.py
and loaded separately at startup.
"""

import numpy as np
import pandas as pd


# ── Missing value handlers ────────────────────────────────────────────────────

def fill_missing_mean(df, column, params):
    """Fill nulls with the column mean (numerical columns only)."""
    df[column] = df[column].fillna(df[column].mean())
    return df


def fill_missing_median(df, column, params):
    """Fill nulls with the column median (numerical columns only)."""
    df[column] = df[column].fillna(df[column].median())
    return df


def fill_missing_mode(df, column, params):
    """Fill nulls with the most frequent value."""
    mode = df[column].mode()
    if len(mode):
        df[column] = df[column].fillna(mode[0])
    return df


def fill_missing_constant(df, column, params):
    """
    Fill nulls with a constant.
    params: {"value": <any>}
    """
    df[column] = df[column].fillna(params.get("value", "UNKNOWN"))
    return df


def fill_missing_ffill(df, column, params):
    """Forward-fill nulls (useful for time-series)."""
    df[column] = df[column].ffill()
    return df


def fill_missing_bfill(df, column, params):
    """Backward-fill nulls."""
    df[column] = df[column].bfill()
    return df


def flag_and_keep(df, column, params):
    """
    Add a boolean companion column <column>_is_missing instead of imputing.
    Preserves original nulls for downstream handling.
    """
    df[column + "_is_missing"] = df[column].isna()
    return df


# ── Drop operations ───────────────────────────────────────────────────────────

def drop_column(df, column, params):
    """Drop the entire column."""
    if column in df.columns:
        df = df.drop(columns=[column])
    return df


def drop_duplicates(df, column, params):
    """
    Drop fully duplicate rows from the dataframe.
    column arg is ignored (operates on whole df).
    params: {"keep": "first" | "last" | false}  — default "first"
    """
    keep = params.get("keep", "first")
    if keep is False or keep == "false":
        keep = False
    before = len(df)
    df = df.drop_duplicates(keep=keep)
    print(f"      drop_duplicates: removed {before - len(df)} rows")
    return df


# ── Outlier / distribution handlers ──────────────────────────────────────────

def clip_outliers_iqr(df, column, params):
    """
    Clip outliers to [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    params: {"multiplier": 1.5}  — optional
    """
    multiplier = params.get("multiplier", 1.5)
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    df[column] = df[column].clip(lower, upper)
    return df


def clip_outliers_zscore(df, column, params):
    """
    Clip values beyond z-score threshold.
    params: {"threshold": 3.0}
    """
    threshold = params.get("threshold", 3.0)
    mean = df[column].mean()
    std  = df[column].std()
    lower = mean - threshold * std
    upper = mean + threshold * std
    df[column] = df[column].clip(lower, upper)
    return df


def log_transform(df, column, params):
    """Apply log1p transform to reduce right skew. Handles zeros safely."""
    df[column] = df[column].apply(lambda x: np.log1p(x) if pd.notna(x) and x >= 0 else x)
    return df


def sqrt_transform(df, column, params):
    """Apply sqrt transform to reduce right skew. Handles negatives safely."""
    df[column] = df[column].apply(lambda x: np.sqrt(x) if pd.notna(x) and x >= 0 else x)
    return df


def boxcox_transform(df, column, params):
    """
    Apply Box-Cox transform (requires all positive values).
    Falls back to log1p if scipy unavailable.
    """
    try:
        from scipy.stats import boxcox
        valid = df[column].dropna()
        if (valid <= 0).any():
            print(f"      boxcox: non-positive values found, falling back to log1p")
            return log_transform(df, column, params)
        transformed, _ = boxcox(valid)
        df.loc[df[column].notna(), column] = transformed
    except ImportError:
        print("      boxcox: scipy not installed, falling back to log1p")
        return log_transform(df, column, params)
    return df


# ── Categorical handlers ──────────────────────────────────────────────────────

def merge_categories(df, column, params):
    """
    Merge near-duplicate category labels into one canonical label.
    params: {"from": ["male", "Male", "MALE"], "to": "Male"}
    """
    from_values = params.get("from", [])
    to_value    = params.get("to",   "")
    if not from_values or not to_value:
        print(f"      merge_categories: missing 'from' or 'to' in params")
        return df
    mapping = {v: to_value for v in from_values}
    df[column] = df[column].replace(mapping)
    return df


def to_uppercase(df, column, params):
    """Convert all string values to uppercase."""
    df[column] = df[column].astype(str).str.upper()
    return df


def to_lowercase(df, column, params):
    """Convert all string values to lowercase."""
    df[column] = df[column].astype(str).str.lower()
    return df


def strip_whitespace(df, column, params):
    """Strip leading and trailing whitespace from string values."""
    df[column] = df[column].astype(str).str.strip()
    return df


def strip_whitespace_internal(df, column, params):
    """Collapse internal whitespace runs to a single space."""
    df[column] = df[column].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    return df


def encode_label(df, column, params):
    """
    Ordinal label encoding: maps unique categories to integers 0..N-1.
    params: {"mapping": {"cat": 0, "dog": 1}}  — optional, auto-generated if absent
    """
    mapping = params.get("mapping")
    if not mapping:
        categories = df[column].dropna().unique()
        mapping = {cat: i for i, cat in enumerate(sorted(str(c) for c in categories))}
    df[column] = df[column].map(mapping)
    return df


# ── Type casting ──────────────────────────────────────────────────────────────

def cast_dtype(df, column, params):
    """
    Cast column to a target dtype.
    params: {"to": "int" | "float" | "str" | "bool" | "datetime"}
    """
    target = params.get("to", "str")
    if target == "datetime":
        fmt = params.get("format", None)
        df[column] = pd.to_datetime(df[column], format=fmt, errors="coerce")
    else:
        try:
            df[column] = df[column].astype(target)
        except (ValueError, TypeError) as e:
            print(f"      cast_dtype: failed to cast '{column}' to {target}: {e}")
    return df


# ── Splitting ─────────────────────────────────────────────────────────────────

def split_column(df, column, params):
    """
    Split a column on a delimiter into two new columns.
    params: {"delimiter": ",", "new_columns": ["col_a", "col_b"], "drop_original": true}
    """
    delimiter    = params.get("delimiter", ",")
    new_cols     = params.get("new_columns", [column + "_0", column + "_1"])
    drop_orig    = params.get("drop_original", True)
    max_splits   = len(new_cols) - 1

    split_df = df[column].astype(str).str.split(delimiter, n=max_splits, expand=True)
    for i, col_name in enumerate(new_cols):
        if i < split_df.shape[1]:
            df[col_name] = split_df[i].str.strip()
        else:
            df[col_name] = None

    if drop_orig:
        df = df.drop(columns=[column])
    return df


# ── Review placeholder ────────────────────────────────────────────────────────

def flag_for_review(df, column, params):
    """
    No-op placeholder. Flagged columns are intercepted by the ReviewGate
    before reaching the router, so this function should never be called directly.
    """
    return df


# ── TOOLS registry ────────────────────────────────────────────────────────────
# This dict is the single source of truth for allowed ops.
# The DeepSeek client sends list(TOOLS.keys()) as the allowed ops list.

TOOLS: dict = {
    # Missing value
    "fill_missing_mean":        fill_missing_mean,
    "fill_missing_median":      fill_missing_median,
    "fill_missing_mode":        fill_missing_mode,
    "fill_missing_constant":    fill_missing_constant,
    "fill_missing_ffill":       fill_missing_ffill,
    "fill_missing_bfill":       fill_missing_bfill,
    "flag_and_keep":            flag_and_keep,
    # Drop
    "drop_column":              drop_column,
    "drop_duplicates":          drop_duplicates,
    # Outliers / distribution
    "clip_outliers_iqr":        clip_outliers_iqr,
    "clip_outliers_zscore":     clip_outliers_zscore,
    "log_transform":            log_transform,
    "sqrt_transform":           sqrt_transform,
    "boxcox_transform":         boxcox_transform,
    # Categorical
    "merge_categories":         merge_categories,
    "to_uppercase":             to_uppercase,
    "to_lowercase":             to_lowercase,
    "strip_whitespace":         strip_whitespace,
    "strip_whitespace_internal":strip_whitespace_internal,
    "encode_label":             encode_label,
    # Type
    "cast_dtype":               cast_dtype,
    # Splitting
    "split_column":             split_column,
    # Review (intercepted before router)
    "flag_for_review":          flag_for_review,
}
