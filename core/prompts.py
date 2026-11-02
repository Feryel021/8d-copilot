"""
prompts.py
----------
Every prompt of the app lives here, so they are easy to read, compare
and grade. Each prompt is labelled with the technique(s) it uses and
follows the 3C principle: Clarity, Context, Constraints.

Techniques used in this project
  1. Zero-shot           -> trend summary, prompt-lab baseline
  2. Few-shot            -> defect classification (5 worked examples)
  3. Chain of Thought    -> 5-Why root-cause analysis
  4. Persona / role      -> senior quality engineer, customer-quality manager
  5. Structured output   -> JSON schemas, fixed 8D template
  6. Iterative refinement / self-check -> 8D review-and-improve step
"""

DEFECT_CATEGORIES = [
    "Wrong wiring / crossed wires",
    "Terminal not seated / back-out",
    "Crimp defect",
    "Missing or wrong component",
    "Damaged insulation / cable",
    "Wrong label / marking",
    "Dimension / length out of tolerance",
    "Packaging / logistics",
    "Other",
]

SEVERITIES = ["High", "Medium", "Low"]

# ==========================================================================
# STEP 1 - Classification   (Persona + FEW-SHOT + JSON constraints)
# Recommended params: temperature 0.0 (we want the same answer every time)
# ==========================================================================
CLASSIFY_SYSTEM = f"""You are a senior automotive quality engineer at a wire-harness
supplier (IATF 16949). You read customer complaints and classify them.

Rules:
- category must be exactly one of: {DEFECT_CATEGORIES}
- severity: High = safety/function risk or line stop at customer,
  Medium = function OK but rework needed, Low = cosmetic or documentation.
- part: the component or area concerned, max 5 words.
- summary: one neutral sentence, max 20 words.
- Answer ONLY with JSON, no extra text."""

# Few-shot examples: input -> expected output
CLASSIFY_EXAMPLES = [
    ("Connector X12 pin 4 and pin 7 swapped, vehicle failed EOL test, line stopped 40 min.",
     {"category": "Wrong wiring / crossed wires", "severity": "High",
      "part": "Connector X12", "summary": "Pins 4 and 7 swapped on X12, causing end-of-line failure and line stop."}),
    ("Terminal pushed out of housing of door connector when operator plugged it.",
     {"category": "Terminal not seated / back-out", "severity": "High",
      "part": "Door connector", "summary": "Terminal backed out of the door connector housing during mating."}),
    ("Label on harness shows old part number revision B instead of C. Function OK.",
     {"category": "Wrong label / marking", "severity": "Low",
      "part": "Harness label", "summary": "Label shows revision B instead of revision C; function not affected."}),
    ("Branch to ABS sensor 30 mm too short, operator had to stretch it to install.",
     {"category": "Dimension / length out of tolerance", "severity": "Medium",
      "part": "ABS sensor branch", "summary": "ABS sensor branch 30 mm too short, installation required stretching."}),
    ("Insulation cut near the grommet, copper visible, found at incoming inspection.",
     {"category": "Damaged insulation / cable", "severity": "High",
      "part": "Cable near grommet", "summary": "Insulation cut near grommet with exposed copper, found at incoming inspection."}),
]


def build_classify_messages(complaint: str) -> list[dict]:
    """Few-shot: examples are given as earlier user/assistant turns."""
    import json
    msgs = [{"role": "system", "content": CLASSIFY_SYSTEM}]
    for text, answer in CLASSIFY_EXAMPLES:
        msgs.append({"role": "user", "content": f"Complaint: {text}"})
        msgs.append({"role": "assistant", "content": json.dumps(answer)})
    msgs.append({"role": "user", "content": f"Complaint: {complaint}"})
    return msgs


# ==========================================================================
# STEP 2 - Trend summary   (ZERO-SHOT + Persona + constraints)
# Recommended params: temperature 0.3
# ==========================================================================
def build_trend_messages(pareto_table: str, n: int) -> list[dict]:
    return [
        {"role": "system", "content": "You are a quality manager preparing a short management-review note."},
        {"role": "user", "content": f"""Below is a Pareto of {n} customer complaints by defect category
and severity (already counted by the software):

{pareto_table}

Write a summary for the plant manager:
- 3 bullet points on the main trends (mention numbers)
- 1 bullet: which category to attack first and why (Pareto logic, severity)
- Max 120 words. Plain English.
- Only quote numbers that appear in the table. Do not add, combine or
  recalculate numbers yourself."""},
    ]


# ==========================================================================
# STEP 3 - Root cause   (Persona + CHAIN OF THOUGHT + JSON)
# Recommended params: temperature 0.4 (some creativity for hypotheses)
# ==========================================================================
RCA_SYSTEM = """You are a senior quality engineer leading an 8D problem-solving team
in a wire-harness plant. You are careful: you separate facts from hypotheses."""


def build_rca_messages(complaint: str, category: str, context: str = "") -> list[dict]:
    extra = f"\nExtra information from the plant: {context}" if context.strip() else ""
    return [
        {"role": "system", "content": RCA_SYSTEM},
        {"role": "user", "content": f"""Complaint: {complaint}
Category: {category}{extra}

Think step by step before concluding:
Step 1. List the facts you know from the complaint (no guessing).
Step 2. Use the Ishikawa 6M (Man, Machine, Method, Material, Measurement,
        Environment) and give one plausible cause per M for a harness plant.
Step 3. Pick the most likely cause and run a 5-Why chain on it.
Step 4. Do a separate 5-Why for "why was it NOT DETECTED" (escape point).
Step 5. Propose verification: how would the team confirm the root cause?

Return ONLY JSON with this structure:
{{
 "facts": ["..."],
 "ishikawa": {{"Man": "...", "Machine": "...", "Method": "...",
              "Material": "...", "Measurement": "...", "Environment": "..."}},
 "occurrence_5why": ["Why 1 ...", "Why 2 ...", "Why 3 ...", "Why 4 ...", "Why 5 ..."],
 "detection_5why": ["Why 1 ...", "Why 2 ...", "Why 3 ..."],
 "root_cause_occurrence": "...",
 "root_cause_detection": "...",
 "verification": "...",
 "confidence": "Low | Medium | High"
}}"""},
    ]


# ==========================================================================
# STEP 4 - 8D report   (Persona + structured template + constraints)
# Recommended params: temperature 0.5, max_tokens ~1500
# ==========================================================================
def build_8d_messages(complaint: str, classification: dict, rca: dict, team: str) -> list[dict]:
    import json
    return [
        {"role": "system", "content": RCA_SYSTEM + " You write clear, audit-ready 8D reports."},
        {"role": "user", "content": f"""Write an 8D report draft in Markdown.

Complaint: {complaint}
Classification: {json.dumps(classification)}
Root-cause analysis: {json.dumps(rca)}
Team: {team or "to be defined"}

Use exactly these headings:
## D1 Team
## D2 Problem description (5W2H)
## D3 Containment actions
## D4 Root cause (occurrence and non-detection)
## D5 Permanent corrective actions
## D6 Implementation and validation
## D7 Prevent recurrence (PFMEA, control plan, lessons learned)
## D8 Closure and team recognition

Constraints:
- D3, D5, D6 as Markdown tables with columns: Action | Owner | Due | Status
- Use "TBD" for owners and dates you do not know. Never invent names.
- Mark anything not yet confirmed as "(hypothesis - to verify)".
- Max 600 words."""},
    ]


# ==========================================================================
# STEP 5 - Self-review   (ITERATIVE REFINEMENT: the model checks its draft)
# Recommended params: temperature 0.2
# ==========================================================================
def build_review_messages(report_md: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are a strict customer quality auditor (VDA 6.3 / IATF 16949)."},
        {"role": "user", "content": f"""Review this 8D draft and improve it.

Check:
1. Is the root cause specific (not "operator error")?
2. Is there a non-detection root cause?
3. Does every corrective action match a root cause?
4. Is D7 updating PFMEA and control plan?

First write a section "## Auditor remarks" with max 4 bullets.
Then write "## Improved 8D" with the full corrected report (same headings).

Draft:
{report_md}"""},
    ]


# ==========================================================================
# STEP 6 - Customer reply email   (Persona + tone constraints)
# Recommended params: temperature 0.7 (natural wording)
# ==========================================================================
def build_email_messages(complaint: str, rca: dict, tone: str) -> list[dict]:
    import json
    return [
        {"role": "system", "content": "You are the customer quality manager of a wire-harness supplier."},
        {"role": "user", "content": f"""Write a short email to the customer about this complaint.

Complaint: {complaint}
Our analysis so far: {json.dumps(rca)}
Tone: {tone}

Include: acknowledgement, containment already started, next step and
date "TBD" for the full 8D. Max 150 words. Subject line first.
Do not admit legal liability. Do not invent numbers."""},
    ]


# ==========================================================================
# PROMPT LAB - the same task with 3 techniques, for experimentation
# ==========================================================================
def lab_zero_shot(complaint: str) -> list[dict]:
    return [{"role": "user", "content":
             f"Classify this wire-harness complaint by defect type and severity:\n{complaint}"}]


def lab_few_shot(complaint: str) -> list[dict]:
    return build_classify_messages(complaint)


def lab_cot(complaint: str) -> list[dict]:
    return [
        {"role": "system", "content": CLASSIFY_SYSTEM.replace("Answer ONLY with JSON, no extra text.", "")},
        {"role": "user", "content": f"""Complaint: {complaint}

Think step by step:
1. What exactly failed?  2. Where was it found?  3. What is the risk for the vehicle?
Then give the final answer as JSON with category, severity, part, summary."""},
    ]


LAB_TECHNIQUES = {
    "Zero-shot (no examples, no rules)": lab_zero_shot,
    "Few-shot + persona + JSON rules": lab_few_shot,
    "Chain of Thought": lab_cot,
}
