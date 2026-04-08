# Dataset Cleaning Agent - Working Progress

An automated dataset cleaning pipeline that combines a statistical auditor,
DeepSeek LLM recommendations, pre-built pandas tools, and a human-in-the-loop
review gate for out-of-scope problems.

---

## Architecture - Plan

```
DatasetAuditor  →  audit_report.json
                         │
                         ▼
              DeepSeek API (per column)
                         │
              ┌──────────┴──────────┐
              │                     │
         Standard ops         flag_for_review
         (Deterministic)      (Non-Deterministic)
              │                     │
          Op Router            Review Gate ← Human Gate 1
              │                     │
         TOOLS dict           LLM/SLM writes tool
              │                     │
         df modified          Human Gate 2 (approve code)
                                    │
                              register to TOOLS
                              persist to custom_tools.py
                                    │
                               df modified
```

---
## Progress

1. Audit report generator build and currently being ammended
2. Pi Mono Agent tested separately

## Setup (Current Setup is Based on ProtoType with LangChain)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your DeepSeek API key

```bash
export DEEPSEEK_API_KEY="your-key-here"
```

Get a key at https://platform.deepseek.com

### 3. Place `dataset_audit_report.py` in the project root

The `DatasetAuditor` class from your audit pipeline must be importable.
Put `dataset_audit_report.py` in the same directory as `main.py`.

---

## Usage

### Basic run

```bash
python main.py --input data.csv
```

### Dry-run (preview only, no files written)

```bash
python main.py --input data.csv --dry-run
```

### Skip re-running the audit (use existing report)

```bash
python main.py --input data.csv --skip-audit
```

### Change quality score threshold (default: skip columns scoring >= 80)

```bash
python main.py --input data.csv --min-score 70
```

### Full options

```
--input       Path to input dataset (CSV, XLSX, Parquet, JSON)
--output      Path for cleaned output CSV  [default: clean_output.csv]
--report      Path for audit report JSON   [default: audit_report.json]
--dry-run     Preview changes, write nothing
--skip-audit  Use existing audit_report.json
--min-score   Skip columns at or above this quality score [default: 80]
```

---

## Output files

| File | Description |
|------|-------------|
| `clean_output.csv` | The cleaned dataset |
| `audit_report.json` | Raw audit output from DatasetAuditor |
| `change_log.json` | Every operation applied with reasons and confidence |
| `custom_tools.py` | Human-approved custom tools (auto-generated, persists across runs) |

---

## Project structure

```
dataset_cleaner/
├── main.py                     ← Entry point
├── requirements.txt
├── dataset_audit_report.py     ← Place your DatasetAuditor here
├── custom_tools.py             ← Auto-created on first approved custom tool
│
├── core/
│   ├── auditor.py              ← Runs DatasetAuditor, returns parsed report
│   ├── deepseek_client.py      ← Calls DeepSeek API, validates op responses
│   ├── router.py               ← Dispatches ops to pandas tool functions
│   └── change_log.py           ← Records every operation for auditability
│
├── tools/
│   └── builtin_tools.py        ← All pre-built pandas cleaning functions + TOOLS dict
│
└── review/
    └── review_gate.py          ← Human-in-the-loop gate + custom tool generation
```

---

## Allowed operations (op enum)

DeepSeek can only choose from this fixed list. It cannot generate arbitrary code.

| Op | Description |
|----|-------------|
| `fill_missing_mean` | Fill nulls with column mean |
| `fill_missing_median` | Fill nulls with column median |
| `fill_missing_mode` | Fill nulls with most frequent value |
| `fill_missing_constant` | Fill nulls with a constant (`params.value`) |
| `fill_missing_ffill` | Forward-fill nulls |
| `fill_missing_bfill` | Backward-fill nulls |
| `flag_and_keep` | Add `<col>_is_missing` boolean column, keep original |
| `drop_column` | Drop the entire column |
| `drop_duplicates` | Drop fully duplicate rows |
| `clip_outliers_iqr` | Clip outliers to IQR bounds |
| `clip_outliers_zscore` | Clip outliers beyond z-score threshold |
| `log_transform` | log1p transform for right-skewed columns |
| `sqrt_transform` | sqrt transform for right-skewed columns |
| `boxcox_transform` | Box-Cox transform (requires positive values) |
| `merge_categories` | Merge near-duplicate labels into one canonical value |
| `to_uppercase` | Uppercase all string values |
| `to_lowercase` | Lowercase all string values |
| `strip_whitespace` | Strip leading/trailing whitespace |
| `strip_whitespace_internal` | Collapse internal whitespace runs |
| `encode_label` | Ordinal encode categorical column |
| `cast_dtype` | Cast column to target dtype (`params.to`) |
| `split_column` | Split column on delimiter into new columns |
| `flag_for_review` | Problem is out of scope → triggers Review Gate |

---

## Adding new built-in ops

1. Write a function in `tools/builtin_tools.py` with signature `func(df, column, params) -> df`
2. Add it to the `TOOLS` dict at the bottom of that file
3. It will automatically be included in DeepSeek's allowed ops list on the next run

---

## Custom tools

When a column is flagged for review and you approve a DeepSeek-generated tool:

- The function is `exec()`'d into memory for the current run
- It is appended to `custom_tools.py` in the project root
- On future runs it is loaded automatically at startup and added to `TOOLS`
- It behaves identically to built-in tools after that

---

## Quality score interpretation

| Score | Status | Meaning |
|-------|--------|---------|
| 80–100 | GOOD | Skipped by default |
| 50–79 | WARN | Reviewed by DeepSeek |
| 0–49 | BAD | Reviewed by DeepSeek, prioritised first |
