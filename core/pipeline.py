"""
pipeline.py
-----------
The multi-step processing chain. The UI calls these functions; they
call the model with the right prompt and the right parameters.

  complaints --> [1 classify] --> [2 Pareto + trend summary]
             --> [3 root cause (CoT)] --> [4 8D draft] --> [5 self-review]
             --> [6 customer email]

Each step has its own recommended parameters (see STEP_PARAMS).
Low temperature where we need consistency (classification), higher
where we want natural text (email).
"""

from __future__ import annotations

import pandas as pd

from . import prompts as P
from .llm_client import GenParams, LLMClient, LLMError

# max_tokens includes the model's hidden "thinking" tokens on Gemini 3,
# so the budgets are larger than the visible answer needs.
STEP_PARAMS = {
    "classify": GenParams(temperature=0.0, max_tokens=1500, top_p=1.0),
    "trend":    GenParams(temperature=0.3, max_tokens=2000, top_p=0.9),
    "rca":      GenParams(temperature=0.4, max_tokens=4000, top_p=0.9),
    "report":   GenParams(temperature=0.5, max_tokens=6000, top_p=0.9),
    "review":   GenParams(temperature=0.2, max_tokens=7000, top_p=0.9),
    "email":    GenParams(temperature=0.7, max_tokens=2000, top_p=0.95),
}


def _validate_classification(c: dict) -> dict:
    """The model can still answer outside the allowed list. Fix it here."""
    if not isinstance(c, dict):
        raise LLMError("Classification was not a JSON object.")
    if c.get("category") not in P.DEFECT_CATEGORIES:
        c["category"] = "Other"
    if c.get("severity") not in P.SEVERITIES:
        c["severity"] = "Medium"
    c.setdefault("part", "unknown")
    c.setdefault("summary", "")
    return c


def classify_all(client: LLMClient, complaints: list[str], params: GenParams | None = None,
                 progress=None) -> pd.DataFrame:
    """Step 1. One call per complaint. A failure on one row does not stop the rest."""
    params = params or STEP_PARAMS["classify"]
    rows = []
    for i, text in enumerate(complaints, start=1):
        try:
            c = client.chat_json(P.build_classify_messages(text), params,
                                 step="1 Classify", technique="Few-shot + persona + JSON")
            c = _validate_classification(c)
            c["error"] = ""
        except LLMError as e:
            # authentication / model errors affect every row: stop now
            msg = str(e)
            if "API key" in msg or "not found" in msg or "no access" in msg:
                raise
            c = {"category": "Other", "severity": "Medium", "part": "-",
                 "summary": "", "error": msg}
        c["complaint"] = text
        rows.append(c)
        if progress:
            progress(i / len(complaints), f"Classified {i} of {len(complaints)}")
    df = pd.DataFrame(rows)
    df.insert(0, "id", range(1, len(df) + 1))
    return df[["id", "complaint", "category", "severity", "part", "summary", "error"]]


def pareto(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2a. Pure Python counting, the model never invents the numbers."""
    ok = df[df["error"] == ""]
    table = (ok.pivot_table(index="category", columns="severity", values="id",
                            aggfunc="count", fill_value=0))
    for s in P.SEVERITIES:
        if s not in table.columns:
            table[s] = 0
    table = table[P.SEVERITIES]
    table["Total"] = table.sum(axis=1)
    table = table.sort_values("Total", ascending=False)
    table["Cumulative %"] = (table["Total"].cumsum() / table["Total"].sum() * 100).round(1)
    return table


def trend_summary(client: LLMClient, pareto_df: pd.DataFrame, n: int,
                  params: GenParams | None = None) -> str:
    """Step 2b. Zero-shot summary of the Pareto table."""
    return client.chat(P.build_trend_messages(pareto_df.to_string(), n),
                       params or STEP_PARAMS["trend"], step="2 Trend summary",
                       technique="Zero-shot + persona")


def root_cause(client: LLMClient, complaint: str, category: str, context: str = "",
               params: GenParams | None = None) -> dict:
    """Step 3. Chain-of-thought 5-Why + Ishikawa."""
    rca = client.chat_json(P.build_rca_messages(complaint, category, context),
                           params or STEP_PARAMS["rca"], step="3 Root cause",
                           technique="Chain of Thought + persona + JSON")
    if not isinstance(rca, dict) or "occurrence_5why" not in rca:
        raise LLMError("Root-cause answer is missing the 5-Why chain. Try again.")
    return rca


def eight_d(client: LLMClient, complaint: str, classification: dict, rca: dict,
            team: str = "", self_review: bool = True,
            params: GenParams | None = None) -> tuple[str, str | None]:
    """Steps 4 and 5. Draft, then let a 'strict auditor' improve it."""
    draft = client.chat(P.build_8d_messages(complaint, classification, rca, team),
                        params or STEP_PARAMS["report"], step="4 8D draft",
                        technique="Persona + template constraints")
    if not self_review:
        return draft, None
    reviewed = client.chat(P.build_review_messages(draft), STEP_PARAMS["review"],
                           step="5 Self-review", technique="Iterative refinement")
    return draft, reviewed


def customer_email(client: LLMClient, complaint: str, rca: dict, tone: str,
                   params: GenParams | None = None) -> str:
    """Step 6."""
    return client.chat(P.build_email_messages(complaint, rca, tone),
                       params or STEP_PARAMS["email"], step="6 Customer email",
                       technique="Persona + tone constraints")
