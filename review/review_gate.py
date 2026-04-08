"""
review/review_gate.py
---------------------
Human-in-the-loop gate for columns flagged by DeepSeek as out-of-scope.

Two approval gates:
  Gate 1 — Show the column problem, ask: write custom tool? skip?
  Gate 2 — Show generated code, ask: apply it? reject?

Approved custom tools are:
  - exec()'d into memory for this run
  - appended to custom_tools.py so they persist for future runs
"""

import os
import textwrap
import pandas as pd

from core.change_log import ChangeLog

CUSTOM_TOOLS_FILE = "custom_tools.py"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL    = "deepseek-chat"


class ReviewGate:
    def __init__(self, tools: dict, changelog: ChangeLog, dry_run: bool = False):
        self.tools     = tools
        self.changelog = changelog
        self.dry_run   = dry_run

    # ── Public entry point ────────────────────────────────────────

    def handle(self, df: pd.DataFrame, col_report: dict, flag_op: dict) -> pd.DataFrame:
        """
        Called when DeepSeek returns flag_for_review for a column.
        Walks through Gate 1 and optionally Gate 2.
        Returns df (possibly modified if a custom tool was applied).
        """
        column = col_report.get("column", "?")
        reason = flag_op.get("params", {}).get("reason", "No reason provided.")

        self._print_banner(column, col_report, reason)

        if self.dry_run:
            print("  [dry-run] Skipping interactive review in dry-run mode.")
            return df

        # ── Gate 1: decide what to do ─────────────────────────────
        choice = self._gate1_prompt()

        if choice == "2":
            print(f"  [review] Column '{column}' skipped. Logged.")
            self.changelog.record(
                "flag_for_review", column, {},
                f"Human skipped: {reason}", 1.0
            )
            return df

        if choice == "3":
            print(f"  [review] Column '{column}' kept as-is.")
            return df

        # choice == "1" → write custom tool
        df = self._gate2_write_tool(df, col_report, reason)
        return df

    # ── Gate 1 ────────────────────────────────────────────────────

    def _gate1_prompt(self) -> str:
        print()
        print("  Options:")
        print("    [1] Ask DeepSeek to write a custom tool for this column")
        print("    [2] Skip this column (log it for later)")
        print("    [3] Keep column as-is (no change)")
        print()
        while True:
            choice = input("  Your choice (1/2/3): ").strip()
            if choice in ("1", "2", "3"):
                return choice
            print("  Please enter 1, 2, or 3.")

    # ── Gate 2 ────────────────────────────────────────────────────

    def _gate2_write_tool(
        self, df: pd.DataFrame, col_report: dict, reason: str
    ) -> pd.DataFrame:
        column = col_report.get("column", "?")
        print(f"\n  [review] Asking DeepSeek to write a custom tool for '{column}'...")

        code_str  = self._generate_custom_tool(col_report, reason)
        func_name = f"custom_{column.lower().replace(' ', '_')}"

        if not code_str:
            print("  [review] DeepSeek could not generate a tool. Skipping column.")
            return df

        # Show the generated code
        print("\n" + "─"*60)
        print(f"  GENERATED TOOL: {func_name}")
        print("─"*60)
        print(code_str)
        print("─"*60)

        # Ask for approval
        print()
        print("  [1] Apply this tool to the column")
        print("  [2] Reject — skip this column")
        print()
        while True:
            choice = input("  Approve code? (1/2): ").strip()
            if choice in ("1", "2"):
                break
            print("  Please enter 1 or 2.")

        if choice == "2":
            print(f"  [review] Tool rejected. Column '{column}' skipped.")
            self.changelog.record(
                "custom_tool_rejected", column, {},
                f"Human rejected generated tool: {reason}", 1.0
            )
            return df

        # Register and apply
        fn = self._register_tool(code_str, func_name)
        if fn is None:
            print(f"  [review] Could not register tool. Column '{column}' skipped.")
            return df

        try:
            before_shape = df.shape
            df = fn(df, column, {})
            after_shape  = df.shape
            print(f"  [review] Custom tool applied. shape {before_shape} -> {after_shape}")
            self.changelog.record(
                func_name, column, {},
                f"Custom tool (human approved): {reason}", 1.0
            )
        except Exception as e:
            print(f"  [review] Custom tool raised an error: {e}")
            self.changelog.record(
                func_name, column, {},
                f"Custom tool failed at runtime: {e}", 1.0,
                error=str(e)
            )

        return df

    # ── Tool generation via DeepSeek ──────────────────────────────

    def _generate_custom_tool(self, col_report: dict, reason: str) -> str | None:
        try:
            from openai import OpenAI
        except ImportError:
            print("  [error] openai package required: pip install openai")
            return None

        import os
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            print("  [error] DEEPSEEK_API_KEY not set.")
            return None

        column = col_report.get("column", "COLUMN")
        func_name = f"custom_{column.lower().replace(' ', '_')}"

        system_prompt = """You are a senior Python data engineer.
Write a single pandas cleaning function for the column described.

STRICT RULES:
1. Output ONLY the Python function definition — no explanation, no markdown fences.
2. Function signature must be exactly:
       def {func_name}(df, column, params):
3. Must return the modified df.
4. Use only pandas and numpy (already imported as pd and np in the execution scope).
5. No file I/O, no network calls, no subprocess, no exec/eval.
6. Handle NaN values safely.
7. Add a one-line docstring describing what the function does.
""".replace("{func_name}", func_name)

        import json
        user_prompt = (
            f"Column: {col_report.get('column')}\n"
            f"Type: {col_report.get('type')}\n"
            f"Problem: {reason}\n"
            f"Sample values: {json.dumps(col_report.get('random_sample', []))}\n"
            f"Quality score: {col_report.get('quality_score')}/100\n"
            f"Recommendations: {json.dumps(col_report.get('recommendations', []))}"
        )

        client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown fences if model adds them
            import re
            raw = re.sub(r"^```(?:python)?", "", raw).strip()
            raw = re.sub(r"```$",            "", raw).strip()
            return raw
        except Exception as e:
            print(f"  [error] DeepSeek code generation failed: {e}")
            return None

    # ── Tool registration ─────────────────────────────────────────

    def _register_tool(self, code_str: str, func_name: str):
        """
        exec() the code into a namespace, register into TOOLS dict,
        and persist to custom_tools.py for future runs.
        Returns the callable or None on failure.
        """
        import numpy as np
        import pandas as pd

        namespace = {"pd": pd, "np": np}
        try:
            exec(code_str, namespace)
        except SyntaxError as e:
            print(f"  [register] Syntax error in generated code: {e}")
            return None

        fn = namespace.get(func_name)
        if fn is None:
            # Try to find any callable that was defined
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_") and name not in ("pd", "np"):
                    fn = obj
                    func_name = name
                    break

        if fn is None:
            print(f"  [register] Could not find function '{func_name}' in generated code.")
            return None

        # Register in live TOOLS dict
        self.tools[func_name] = fn
        print(f"  [register] '{func_name}' registered in TOOLS.")

        # Persist to custom_tools.py
        self._persist_tool(code_str, func_name)
        return fn

    def _persist_tool(self, code_str: str, func_name: str):
        """Append approved tool to custom_tools.py."""
        header = (
            "\n\n# " + "─"*56 + "\n"
            f"# Auto-generated and human-approved: {func_name}\n"
            "# " + "─"*56 + "\n"
        )
        # Ensure imports are at the top of the file
        if not os.path.exists(CUSTOM_TOOLS_FILE):
            with open(CUSTOM_TOOLS_FILE, "w") as f:
                f.write("import pandas as pd\nimport numpy as np\n")

        with open(CUSTOM_TOOLS_FILE, "a") as f:
            f.write(header + textwrap.dedent(code_str) + "\n")

        print(f"  [persist] Tool saved to {CUSTOM_TOOLS_FILE}")

    # ── Helpers ───────────────────────────────────────────────────

    def _print_banner(self, column: str, col_report: dict, reason: str):
        score  = col_report.get("quality_score", "?")
        sample = col_report.get("random_sample", [])
        print()
        print("  " + "═"*56)
        print(f"  REVIEW REQUIRED: {column}  (score={score}/100)")
        print("  " + "─"*56)
        print(f"  Reason:  {reason}")
        print(f"  Sample:  {sample}")
        print("  " + "═"*56)
