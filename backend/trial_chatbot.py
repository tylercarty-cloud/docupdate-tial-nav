"""
Trial Finder Chatbot — Helps doctors find Gainesville clinical trials and draft
referral emails to study contacts.
"""
import os
import csv

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

GROQ_MODEL = "llama-3.3-70b-versatile"  # or "llama-3.1-8b-instant" for faster/cheaper

load_dotenv()

CSV_PATH = "gainesville_active_trials.csv"
TOP_K = 8
MAX_CRITERIA_LEN = 1500


def load_trials() -> pd.DataFrame:
    """Load trials CSV, handling multiline fields."""
    if not os.path.exists(CSV_PATH):
        st.error(f"Trial data not found at {CSV_PATH}. Run the fetch script first.")
        st.stop()
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return pd.DataFrame(rows)


@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def build_trial_index(df: pd.DataFrame):
    def row_text(r):
        parts = [
            str(r.get("title", "")),
            str(r.get("conditions", "")),
            str(r.get("brief_summary", ""))[:800],
            str(r.get("interventions", "")),
            str(r.get("primary_outcomes", ""))[:300],
        ]
        return " | ".join(p for p in parts if p and p != "nan")

    texts = [row_text(row) for _, row in df.iterrows()]
    embedder = get_embedder()
    embeddings = embedder.encode(texts, show_progress_bar=True)
    return np.array(embeddings, dtype=np.float32)


def search_trials(df: pd.DataFrame, embeddings: np.ndarray, query: str, top_k: int = TOP_K) -> pd.DataFrame:
    embedder = get_embedder()
    q_emb = embedder.encode([query])
    scores = np.dot(embeddings, q_emb.T).flatten()
    idx = np.argsort(scores)[::-1][:top_k]
    return df.iloc[idx].copy()


def trial_to_context(row: pd.Series) -> str:
    def trunc(s, n=MAX_CRITERIA_LEN):
        s = str(s) if pd.notna(s) else ""
        if s == "nan":
            s = ""
        return (s[:n] + "...") if len(s) > n else s

    inc = trunc(row.get("inclusion_criteria"), MAX_CRITERIA_LEN)
    exc = trunc(row.get("exclusion_criteria"), MAX_CRITERIA_LEN)
    return f"""## {row.get('nct_id', '')} — {row.get('title', '')}
- **Conditions:** {row.get('conditions', '')}
- **Locations:** {row.get('locations', '')}
- **Contacts:** {row.get('contacts', '')}
- **Gainesville recruiting:** {row.get('gainesville_active_recruiting', '')}
- **Phase:** {row.get('phase', '')} | **Enrollment:** {row.get('enrollment', '')} | **Sponsor:** {row.get('sponsor', '')}
- **Sex:** {row.get('sex', '')} | **Age:** {row.get('minimum_age', '')}–{row.get('maximum_age', '')}
- **Brief summary:** {trunc(row.get('brief_summary', ''), 600)}
- **Interventions:** {row.get('interventions', '')}
- **Inclusion criteria:** {inc}
- **Exclusion criteria:** {exc}
"""


def draft_referral_email(row: pd.Series, patient_summary: str) -> str:
    contacts = str(row.get("contacts", ""))
    title = str(row.get("title", ""))
    nct_id = str(row.get("nct_id", ""))
    conditions = str(row.get("conditions", ""))

    client = Groq()
    prompt = f"""You are helping a physician draft a professional referral email to a clinical trial coordinator.

**Trial:** {title}
**NCT ID:** {nct_id}
**Conditions:** {conditions}
**Study contact(s):** {contacts}

**Patient summary (from the physician):**
{patient_summary}

Write a concise, professional email that:
1. Introduces the physician and states the purpose (referring a potential participant)
2. Briefly summarizes the patient's relevant history and why they may be a good fit
3. Asks to discuss next steps (screening, enrollment)
4. Includes the physician's contact info placeholder [Your name, credentials, phone, email]
5. Is respectful of the coordinator's time and suitable for a medical professional

Use a professional tone. Include a suggested subject line at the very top, then a blank line, then the email body.
Format as plain text suitable for copy-paste into an email client."""

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


SYSTEM_PROMPT = """You are a clinical trial assistant for physicians. You help them find the best Gainesville, FL trials for their patients.

You have access to trial data from ClinicalTrials.gov. When the physician describes a patient, use the provided trial information to:
1. Identify trials that may be a good fit based on condition, eligibility (age, sex, inclusion/exclusion)
2. Prioritize trials where Gainesville is actively recruiting
3. Clearly explain why each trial might or might not fit
4. Cite NCT IDs (e.g., NCT01234567) so the physician can look up full details

If the physician asks to "draft an email" or "write a referral email" for a specific trial, tell them to click the "Draft email" button next to that trial—the app will generate it.

Be concise and clinically relevant. Do not give medical advice—you assist with trial matching only."""


def main():
    st.set_page_config(page_title="Trial Finder for Physicians", page_icon="🔬", layout="wide")
    st.title("🔬 Trial Finder")
    st.caption("Find Gainesville clinical trials and draft referral emails to study contacts.")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.warning("Set `GROQ_API_KEY` in a `.env` file to use the chatbot.")
        st.info("Create a `.env` file with: `GROQ_API_KEY=gsk_your-key` (get one at console.groq.com)")
        st.stop()

    with st.spinner("Loading trial data..."):
        df = load_trials()
    st.sidebar.success(f"Loaded {len(df)} trials")

    with st.spinner("Building search index (one-time)..."):
        embeddings = build_trial_index(df)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**How to use**")
    st.sidebar.markdown("1. Describe your patient (condition, age, sex, key history)")
    st.sidebar.markdown("2. Review the matched trials")
    st.sidebar.markdown("3. Click **Draft email** next to a trial to generate a referral email")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "draft" not in st.session_state:
        st.session_state.draft = None

    # Show draft email modal if one was requested
    if st.session_state.draft:
        d = st.session_state.draft
        with st.expander(f"📧 Email draft for {d['nct_id']} — Copy and send", expanded=True):
            st.text_area("Draft (copy to your email client)", d["email"], height=280, key="draft_area")
            if st.button("Close draft"):
                st.session_state.draft = None
                st.rerun()

    for mi, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("trials") is not None and msg["role"] == "assistant":
                trials_df = msg["trials"]
                patient = msg.get("last_patient_summary", "")
                with st.expander("Referenced trials — Draft email", expanded=False):
                    for i, (_, t) in enumerate(trials_df.iterrows()):
                        nct = str(t.get("nct_id", ""))
                        title_short = (str(t.get("title", "")) or "")[:90] + ("..." if len(str(t.get("title", ""))) > 90 else "")
                        st.markdown(f"**{nct}** — {title_short}")
                        st.caption(f"Contacts: {str(t.get('contacts', ''))[:100]}")
                        if st.button(f"📧 Draft email for {nct}", key=f"draft_hist_{nct}_{mi}_{i}"):
                            with st.spinner("Drafting..."):
                                email = draft_referral_email(t, patient)
                            st.session_state.draft = {"nct_id": nct, "email": email}
                            st.rerun()

    if prompt := st.chat_input("Describe your patient (condition, age, sex, key history) or ask about trials..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching trials and drafting response..."):
                trials = search_trials(df, embeddings, prompt)
                context = "\n\n".join(trial_to_context(row) for _, row in trials.iterrows())

                client = Groq()
                user_msg = f"""**Relevant trials (from semantic search):**

{context}

---

**Physician's question/patient description:**
{prompt}

Based on the above trial data, help the physician find the best matches. Cite NCT IDs. If they asked to draft an email, direct them to the "Draft email" button."""

                resp = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.3,
                )
                reply = resp.choices[0].message.content

            st.markdown(reply)

            with st.expander("Trials used in this response — Draft email", expanded=True):
                for i, (_, t) in enumerate(trials.iterrows()):
                    nct = str(t.get("nct_id", ""))
                    title_short = (str(t.get("title", "")) or "")[:90] + ("..." if len(str(t.get("title", ""))) > 90 else "")
                    st.markdown(f"**{nct}** — {title_short}")
                    st.caption(f"Contacts: {str(t.get('contacts', ''))[:120]}")
                    if st.button(f"📧 Draft email for {nct}", key=f"draft_new_{nct}_{i}"):
                        with st.spinner("Drafting..."):
                            email = draft_referral_email(t, prompt)
                        st.session_state.draft = {"nct_id": nct, "email": email}
                        st.rerun()

        import time
        st.session_state.messages.append({
            "role": "assistant",
            "content": reply,
            "trials": trials,
            "last_patient_summary": prompt,
            "ts": time.time(),
        })


if __name__ == "__main__":
    main()
