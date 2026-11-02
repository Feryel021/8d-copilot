"""
8D Copilot - AI assistant for automotive customer complaints
============================================================
Run:  streamlit run app.py

Tabs
  1. Analyze complaints   classify many complaints, Pareto, trend summary
  2. Root cause and 8D    CoT 5-Why, 8D draft, auditor self-review, email
  3. Prompt lab           compare zero-shot / few-shot / CoT with your own parameters
  4. History              saved analyses (SQLite)
  5. Prompt log           every API call: prompt, parameters, answer, time
"""

import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from core import file_loader, pipeline, storage
from core import prompts as P
from core.llm_client import PROVIDERS, GenParams, LLMClient, LLMError

st.set_page_config(page_title="8D Copilot", page_icon="🔧", layout="wide")

# ---------------------------------------------------------------- state
for key, default in {
    "logs": [], "df": None, "pareto": None, "trend": None,
    "rca": None, "draft": None, "reviewed": None, "email": None, "selected_id": None,
}.items():
    st.session_state.setdefault(key, default)


def get_secret(name: str) -> str:
    """Key from Streamlit secrets or environment, if the user set one."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, "")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Model settings")
    provider = st.selectbox("Provider", list(PROVIDERS))
    cfg = PROVIDERS[provider]
    api_key = st.text_input("API key", value=get_secret(cfg["key_name"]), type="password",
                            help=f"Or set {cfg['key_name']} in .streamlit/secrets.toml")
    model = st.text_input("Model", value=cfg["default_model"])

    st.divider()
    st.subheader("Generation parameters")
    manual = st.toggle("Override the per-step defaults", value=False,
                       help="Off: each step uses its tuned values (see table below). "
                            "On: every step uses the sliders.")
    temperature = st.slider("temperature", 0.0, 1.5, 0.3, 0.05, disabled=not manual,
                            help="Low = consistent and factual. High = more varied wording.")
    max_tokens = st.slider("max_tokens", 500, 8000, 3000, 250, disabled=not manual,
                           help="Upper limit for the answer length.")
    top_p = st.slider("top_p", 0.1, 1.0, 0.9, 0.05, disabled=not manual,
                      help="Only the most probable words that add up to this share are considered.")

    with st.expander("Per-step defaults"):
        st.dataframe(pd.DataFrame({k: v.as_dict() for k, v in pipeline.STEP_PARAMS.items()}).T,
                     width="stretch")
    st.caption(f"API calls this session: {len(st.session_state.logs)}")


def params_for(step: str) -> GenParams:
    return GenParams(temperature, max_tokens, top_p) if manual else pipeline.STEP_PARAMS[step]


def make_client() -> LLMClient | None:
    try:
        return LLMClient(provider, api_key, model.strip(), logs=st.session_state.logs)
    except LLMError as e:
        st.error(str(e))
        return None


# ---------------------------------------------------------------- header
st.title("8D Copilot")
st.write("Turns customer complaints from a wire-harness plant into a classified Pareto, "
         "a root-cause analysis and an 8D report draft.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Analyze complaints", "Root cause and 8D", "Prompt lab", "History", "Prompt log"])

# ================================================================ TAB 1
with tab1:
    st.subheader("Step 1. Load complaints")
    source = st.radio("Input", ["Paste text", "Upload file", "Use sample data"], horizontal=True)
    complaints = []
    try:
        if source == "Paste text":
            text = st.text_area("One complaint per line", height=180,
                                placeholder="Connector X5 pin 2 empty, found at customer EOL test...")
            if text.strip():
                complaints = file_loader.from_text(text)
        elif source == "Upload file":
            up = st.file_uploader("CSV (column 'complaint'), TXT or PDF", type=["csv", "txt", "pdf"])
            if up:
                complaints = file_loader.load_upload(up)
        else:
            complaints = file_loader.from_csv(os.path.join(os.path.dirname(__file__), "data", "sample_complaints.csv"))
    except file_loader.InputError as e:
        st.warning(str(e))

    if complaints:
        st.caption(f"{len(complaints)} complaints loaded "
                   f"(max {file_loader.MAX_COMPLAINTS}, long texts cut at {file_loader.MAX_CHARS_PER_ITEM} characters).")
        with st.expander("Preview"):
            st.write(pd.DataFrame({"complaint": complaints}))

    if st.button("Classify complaints", type="primary", disabled=not complaints):
        client = make_client()
        if client:
            bar = st.progress(0.0, "Starting")
            try:
                df = pipeline.classify_all(client, complaints, params_for("classify"),
                                           progress=lambda p, msg: bar.progress(p, msg))
                st.session_state.df = df
                st.session_state.pareto = pipeline.pareto(df)
                st.session_state.trend = None
            except LLMError as e:
                st.error(str(e))
            bar.empty()

    df = st.session_state.df
    if df is not None:
        failed = (df["error"] != "").sum()
        if failed:
            st.warning(f"{failed} complaint(s) could not be classified. See the 'error' column.")
        st.subheader("Results")
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button("Download results (CSV)", df.to_csv(index=False).encode("utf-8"),
                           "classified_complaints.csv", "text/csv")

        if not st.session_state.pareto.empty:
            st.subheader("Step 2. Pareto by defect category")
            c1, c2 = st.columns([3, 2])
            with c1:
                st.bar_chart(st.session_state.pareto[P.SEVERITIES], horizontal=True, sort=False,
                             color=["#c0392b", "#e6a23c", "#7f8c8d"])
            with c2:
                st.dataframe(st.session_state.pareto, width="stretch")

            if st.button("Write management summary"):
                client = make_client()
                if client:
                    with st.spinner("Writing summary"):
                        try:
                            st.session_state.trend = pipeline.trend_summary(
                                client, st.session_state.pareto, len(df), params_for("trend"))
                        except LLMError as e:
                            st.error(str(e))
            if st.session_state.trend:
                st.info(st.session_state.trend)

# ================================================================ TAB 2
with tab2:
    df = st.session_state.df
    if df is None or df.empty:
        st.info("Classify complaints in the first tab, then pick one here for a full root-cause analysis.")
    else:
        options = {f"#{r.id} · {r.category} · {r.severity} · {r.complaint[:70]}": r.id
                   for r in df.itertuples()}
        label = st.selectbox("Complaint to analyze", list(options))
        row = df[df["id"] == options[label]].iloc[0]
        if st.session_state.selected_id != row["id"]:
            st.session_state.update(selected_id=row["id"], rca=None, draft=None, reviewed=None, email=None)

        st.write(f"**Complaint:** {row['complaint']}")
        context = st.text_area("Extra plant information (optional)",
                               placeholder="e.g. new operator on station 12, crimp machine changed last week")
        team = st.text_input("8D team (optional)", placeholder="Quality, Production, Engineering, Maintenance")

        if st.button("Step 3. Find root cause", type="primary"):
            client = make_client()
            if client:
                with st.spinner("Running 5-Why and Ishikawa"):
                    try:
                        st.session_state.rca = pipeline.root_cause(
                            client, row["complaint"], row["category"], context, params_for("rca"))
                        st.session_state.draft = st.session_state.reviewed = None
                    except LLMError as e:
                        st.error(str(e))

        rca = st.session_state.rca
        if rca:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Facts**")
                for f in rca.get("facts", []):
                    st.markdown(f"- {f}")
                st.markdown("**Ishikawa (6M)**")
                st.table(pd.DataFrame(rca.get("ishikawa", {}).items(), columns=["M", "Possible cause"]))
            with c2:
                st.markdown("**5-Why: why did it happen?**")
                for i, w in enumerate(rca.get("occurrence_5why", []), 1):
                    st.markdown(f"{i}. {w}")
                st.markdown("**5-Why: why was it not detected?**")
                for i, w in enumerate(rca.get("detection_5why", []), 1):
                    st.markdown(f"{i}. {w}")
            st.success(f"Occurrence root cause: {rca.get('root_cause_occurrence', '-')}")
            st.success(f"Non-detection root cause: {rca.get('root_cause_detection', '-')}")
            st.caption(f"Verification: {rca.get('verification', '-')}  |  "
                       f"Model confidence: {rca.get('confidence', '-')}")

            self_review = st.checkbox("Let a strict auditor review and improve the draft", value=True)
            if st.button("Step 4. Write 8D report"):
                client = make_client()
                if client:
                    with st.spinner("Writing 8D"):
                        try:
                            classification = row[["category", "severity", "part", "summary"]].to_dict()
                            d, r = pipeline.eight_d(client, row["complaint"], classification, rca,
                                                    team, self_review, params_for("report"))
                            st.session_state.draft, st.session_state.reviewed = d, r
                            storage.save(row["complaint"], classification, rca, r or d, model)
                        except LLMError as e:
                            st.error(str(e))

        if st.session_state.draft:
            if st.session_state.reviewed:
                v1, v2 = st.tabs(["Reviewed version", "First draft"])
                with v1:
                    st.markdown(st.session_state.reviewed)
                with v2:
                    st.markdown(st.session_state.draft)
            else:
                st.markdown(st.session_state.draft)
            final = st.session_state.reviewed or st.session_state.draft
            st.download_button("Download 8D (Markdown)", final.encode("utf-8"),
                               f"8D_complaint_{row['id']}_{datetime.now():%Y%m%d}.md", "text/markdown")

            st.divider()
            tone = st.select_slider("Email tone", ["Formal", "Neutral", "Warm"], value="Neutral")
            if st.button("Step 6. Draft customer email"):
                client = make_client()
                if client:
                    with st.spinner("Writing email"):
                        try:
                            st.session_state.email = pipeline.customer_email(
                                client, row["complaint"], rca, tone, params_for("email"))
                        except LLMError as e:
                            st.error(str(e))
            if st.session_state.email:
                st.text_area("Email draft", st.session_state.email, height=220)

# ================================================================ TAB 3
with tab3:
    st.write("Run the same complaint through different prompt techniques and parameters, "
             "and compare the answers side by side.")
    lab_text = st.text_area("Complaint", value="Two terminals in connector X30 were not locked, "
                            "secondary lock open, vehicle had intermittent airbag warning.", height=90)
    chosen = st.multiselect("Techniques", list(P.LAB_TECHNIQUES), default=list(P.LAB_TECHNIQUES))
    temps = st.multiselect("Temperatures to compare", [0.0, 0.3, 0.7, 1.0, 1.3], default=[0.0, 1.0])
    lab_tokens = st.number_input("max_tokens", 500, 8000, 2000, 250)

    if st.button("Run experiment", type="primary", disabled=not (lab_text.strip() and chosen and temps)):
        client = make_client()
        if client:
            results = []
            bar = st.progress(0.0)
            total = len(chosen) * len(temps)
            for i, (tech, t) in enumerate([(a, b) for a in chosen for b in temps], 1):
                try:
                    out = client.chat(P.LAB_TECHNIQUES[tech](lab_text), GenParams(t, lab_tokens, 1.0),
                                      step="Prompt lab", technique=tech)
                except LLMError as e:
                    out = f"ERROR: {e}"
                log = st.session_state.logs[-1] if not out.startswith("ERROR") else None
                results.append({"technique": tech, "temperature": t, "answer": out,
                                "seconds": log.seconds if log else None,
                                "tokens": log.tokens if log else None})
                bar.progress(i / total)
            bar.empty()
            st.session_state.lab = results

    if st.session_state.get("lab"):
        for tech in dict.fromkeys(r["technique"] for r in st.session_state.lab):
            st.markdown(f"#### {tech}")
            cols = st.columns(len(temps) or 1)
            for col, r in zip(cols, [r for r in st.session_state.lab if r["technique"] == tech]):
                with col:
                    st.caption(f"temperature {r['temperature']} · {r['seconds']} s · {r['tokens']} tokens")
                    st.code(r["answer"], language="json", wrap_lines=True)
        lab_df = pd.DataFrame(st.session_state.lab)
        st.download_button("Download experiment (CSV)", lab_df.to_csv(index=False).encode("utf-8"),
                           "prompt_experiment.csv", "text/csv")
        st.caption("What to look for: does the format stay the same? Is the category one of the allowed "
                   "values? Does the answer change between runs at high temperature?")

# ================================================================ TAB 4
with tab4:
    rows = storage.load_all()
    if not rows:
        st.info("No saved analyses yet. Every 8D you write in the second tab is saved here.")
    else:
        hist = pd.DataFrame(rows)
        st.dataframe(hist[["id", "created", "category", "severity", "complaint", "model"]],
                     width="stretch", hide_index=True)
        pick = st.selectbox("Open report", hist["id"])
        st.markdown(hist[hist["id"] == pick]["report"].iloc[0] or "_No report_")
        if st.button("Delete all history"):
            storage.delete_all()
            st.rerun()

# ================================================================ TAB 5
with tab5:
    logs = st.session_state.logs
    if not logs:
        st.info("Every API call will appear here with its prompt, parameters and answer.")
    else:
        st.dataframe(pd.DataFrame([{"step": l.step, "technique": l.technique, **l.params,
                                    "seconds": l.seconds, "tokens": l.tokens} for l in logs]),
                     width="stretch")
        for i, l in enumerate(reversed(logs), 1):
            with st.expander(f"{l.step} · {l.technique} · T={l.params['temperature']}"):
                st.markdown("**Prompt (messages sent)**")
                st.json(l.messages, expanded=False)
                st.markdown("**Answer**")
                st.code(l.output, wrap_lines=True)
        export = json.dumps([l.__dict__ for l in logs], indent=2, ensure_ascii=False)
        st.download_button("Download prompt log (JSON)", export.encode("utf-8"), "prompt_log.json")
