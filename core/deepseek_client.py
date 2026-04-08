"""
core/deepseek_client.py
-----------------------
Calls the DeepSeek API to get cleaning operation recommendations
for a single column report. Returns a validated list of op dicts.

DeepSeek is OpenAI-API-compatible, so we use the openai SDK
pointed at DeepSeek's base URL.

Setup:
    pip install openai
    Set DEEPSEEK_API_KEY environment variable.
"""

import json
import os
import re

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL    = "deepseek-chat"


def _build_system_prompt(allowed_ops: list[str]) -> str:
    ops_list = "\n".join(f"  - {op}" for op in allowed_ops)
    return f"""You are a data cleaning expert. You will receive a JSON audit report for a single dataset column.

Your job is to return a JSON array of cleaning operations to apply to that column.

STRICT RULES:
1. Respond ONLY with a valid JSON array. No prose, no markdown fences, no explanation.
2. Every object in the array must have exactly these fields:
   - "op"        : string — must be one of the allowed ops below
   - "column"    : string — the column name (copy from the report)
   - "params"    : object — operation parameters (can be empty {{}})
   - "confidence": float  — your confidence 0.0 to 1.0
   - "reason"    : string — one sentence explaining why

3. ALLOWED OPS (use ONLY these exact strings):
{ops_list}

4. Use "flag_for_review" when the issue cannot be fixed by any allowed op.
   In that case set params to {{"reason": "explain the problem clearly"}}.

5. If confidence < 0.7, use "flag_for_review" instead.

6. Return an empty array [] if the column needs no cleaning.

Example response:
[
  {{
    "op": "fill_missing_median",
    "column": "AGE",
    "params": {{}},
    "confidence": 0.95,
    "reason": "Numerical column with 12% missing values, median fill is appropriate."
  }}
]"""


def _build_user_prompt(col_report: dict) -> str:
    # Only send what the model needs — keep tokens low
    payload = {
        "column":          col_report.get("column"),
        "type":            col_report.get("type"),
        "quality_score":   col_report.get("quality_score"),
        "missing_pct":     col_report.get("missing_pct"),
        "random_sample":   col_report.get("random_sample"),
        "recommendations": col_report.get("recommendations", []),
    }
    # Include type-specific fields if present
    for key in ("skewness", "has_outliers", "is_distorted",
                "fuzzy_merge_candidates", "should_split", "semantic_pattern"):
        if key in col_report:
            payload[key] = col_report[key]

    return json.dumps(payload, indent=2)


def _parse_response(raw: str) -> list[dict]:
    """Extract JSON array from DeepSeek response, tolerating minor formatting issues."""
    raw = raw.strip()
    # Strip markdown fences if model ignores the instruction
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$",          "", raw).strip()
    return json.loads(raw)


def _validate_ops(ops: list, allowed_ops: list[str]) -> list[dict]:
    """Keep only ops that are in the allowed list and have required fields."""
    valid   = []
    allowed = set(allowed_ops)
    for op in ops:
        if not isinstance(op, dict):
            continue
        if op.get("op") not in allowed:
            print(f"    [warn] DeepSeek returned unknown op '{op.get('op')}' — skipped.")
            continue
        # Ensure required fields exist
        op.setdefault("params",     {})
        op.setdefault("confidence", 0.5)
        op.setdefault("reason",     "")
        valid.append(op)
    return valid


def get_cleaning_ops(col_report: dict, allowed_ops: list[str]) -> list[dict]:
    """
    Call DeepSeek API for a single column report.
    Returns a validated list of op dicts.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package is required: pip install openai"
        )

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "DEEPSEEK_API_KEY environment variable is not set."
        )

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    col_name = col_report.get("column", "unknown")
    print(f"    -> Asking DeepSeek for ops on column: {col_name}")

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system",  "content": _build_system_prompt(allowed_ops)},
                {"role": "user",    "content": _build_user_prompt(col_report)},
            ],
            temperature=0.0,   # deterministic
            max_tokens=1024,
        )
        raw = response.choices[0].message.content
        ops = _parse_response(raw)
        ops = _validate_ops(ops, allowed_ops)
        print(f"    -> {len(ops)} op(s) returned.")
        return ops

    except json.JSONDecodeError as e:
        print(f"    [error] Could not parse DeepSeek response as JSON: {e}")
        print(f"    Raw response: {raw[:300]}")
        return []
    except Exception as e:
        print(f"    [error] DeepSeek API call failed: {e}")
        return []
