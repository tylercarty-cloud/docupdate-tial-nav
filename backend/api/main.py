"""
Trial Finder API — FastAPI backend for chat and email drafting.
Uses stdlib + openai only (no pandas/sklearn) so Vercel serverless stays under bundle limit.
"""
import os
import csv
import re
import math
from collections import Counter
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

_GEO = None  # pgeocode omitted for Vercel bundle size; ZIP_FALLBACK used

load_dotenv()

OPENAI_MODEL = "gpt-4o-mini"
# Primary: multi-location (6 cities); fallback: Gainesville-only
CSV_PATHS = ["active_trials.csv", "gainesville_active_trials.csv"]
TOP_K = 5
MAX_TRIALS_DISPLAY = 3
MAX_CRITERIA_LEN = 400

# Trial site coordinates (city centers) for distance calculation
TRIAL_SITE_COORDS = {
    "Gainesville Florida": (29.6516, -82.3248),
    "Los Angeles California": (34.0522, -118.2437),
    "Dallas Texas": (32.7767, -96.7970),
    "Minneapolis Minnesota": (44.9778, -93.2650),
    "New York New York": (40.7128, -74.0060),
    "Seattle Washington": (47.6062, -122.3321),
}

# (query_locn key, display label) for recruiting-at text and distance logic
LOCATION_LABELS = [
    ("Gainesville Florida", "Gainesville, FL"),
    ("Los Angeles California", "Los Angeles, CA"),
    ("Dallas Texas", "Dallas, TX"),
    ("Minneapolis Minnesota", "Minneapolis, MN"),
    ("New York New York", "New York, NY"),
    ("Seattle Washington", "Seattle, WA"),
]
# Map CSV active_recruiting column suffix to TRIAL_SITE_COORDS key
LOCATION_ACTIVE_COLUMNS = [
    ("gainesville_active_recruiting", "Gainesville Florida"),
    ("los_angeles_active_recruiting", "Los Angeles California"),
    ("dallas_active_recruiting", "Dallas Texas"),
    ("minneapolis_active_recruiting", "Minneapolis Minnesota"),
    ("new_york_active_recruiting", "New York New York"),
    ("seattle_active_recruiting", "Seattle Washington"),
]

# Fallback zip→coords when pgeocode not installed (common areas)
ZIP_FALLBACK = {
    "32601": (29.6516, -82.3248),   # Gainesville, FL
    "32603": (29.6516, -82.3248),
    "90001": (33.9731, -118.2489),  # Los Angeles, CA
    "90012": (34.0522, -118.2437),
    "90210": (34.0901, -118.4065),
    "75201": (32.7767, -96.7970),   # Dallas, TX
    "55401": (44.9778, -93.2650),   # Minneapolis, MN
    "10001": (40.7128, -74.0060),   # New York, NY
    "98101": (47.6062, -122.3321),  # Seattle, WA
}

# Distance brackets in order of importance (miles)
DISTANCE_BRACKETS = [(1, 25), (26, 50), (51, 100), (101, 9999)]

# Globals (loaded at startup or on first request in serverless)
_rows: list[dict] | None = None
_doc_counters: list[Counter] | None = None
_idf: dict | None = None

_STOP = frozenset(
    "a an and are as at be by for from has he in is it its of on that the to was were will with".split()
)


def ensure_loaded() -> None:
    """Load trial data and search index once (for serverless where lifespan may not run)."""
    global _rows, _doc_counters, _idf
    if _rows is not None:
        return
    _rows = load_trials()
    texts = [row_text(r) for r in _rows]
    doc_tokens = [_tokenize(t) for t in texts]
    _doc_counters = [Counter(toks) for toks in doc_tokens]
    doc_freq: Counter = Counter()
    for toks in doc_tokens:
        doc_freq.update(set(toks))
    n_docs = len(_rows) + 1
    _idf = {t: math.log(n_docs / (doc_freq[t] + 1)) + 1 for t in doc_freq}


def _tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in toks if t not in _STOP and len(t) > 1]


def load_trials() -> list[dict]:
    # Try multiple bases so it works from backend/, project root, or Vercel deployment
    api_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.normpath(os.path.join(api_dir, ".."))
    cwd = os.getcwd()
    candidates = []
    for name in CSV_PATHS:
        candidates.append(os.path.join(backend_dir, name))
        candidates.append(os.path.join(cwd, "backend", name))
        candidates.append(os.path.join(cwd, name))
    for path in candidates:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            return [_normalize_row(r) for r in rows]
    raise FileNotFoundError(
        f"Trial data not found. Tried: {CSV_PATHS}. Paths checked: {candidates[:6]}"
    )


def _normalize_row(r: dict) -> dict:
    return {k: _safe_val(v) for k, v in r.items()}


def row_text(r: dict) -> str:
    parts = [
        str(r.get("title", "")),
        str(r.get("conditions", "")),
        (str(r.get("brief_summary", "")) or "")[:800],
        str(r.get("interventions", "")),
        (str(r.get("primary_outcomes", "")) or "")[:300],
        (str(r.get("secondary_outcomes", "")) or "")[:300],
    ]
    return " | ".join(p for p in parts if p and p != "nan")


def search_trials(query: str, top_k: int = TOP_K) -> list[dict]:
    global _rows, _doc_counters, _idf
    if _rows is None or _doc_counters is None or _idf is None:
        ensure_loaded()
    q_toks = _tokenize(query)
    if not q_toks:
        return (_rows or [])[:top_k]
    scored: list[tuple[float, int]] = []
    for idx, doc_cnt in enumerate(_doc_counters or []):
        score = sum(doc_cnt.get(t, 0) * (_idf or {}).get(t, 1) for t in q_toks)
        scored.append((score, idx))
    scored.sort(key=lambda x: -x[0])
    out = []
    rows = _rows or []
    for _, idx in scored[:top_k]:
        out.append(rows[idx])
    return out


def _safe_val(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return ""
    return s


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in miles between two lat/lon points."""
    R = 3959  # Earth radius in miles
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _zip_to_coords(zip_code: str) -> tuple[float, float] | None:
    """Return (lat, lon) for a US zip code, or None if not found."""
    zip_code = (zip_code or "").strip()[:5]
    if not zip_code or not zip_code.isdigit():
        return None
    if zip_code in ZIP_FALLBACK:
        return ZIP_FALLBACK[zip_code]
    if _GEO is None:
        return None
    result = _GEO.query_postal_code(zip_code)
    if result is None or (getattr(result, "latitude", None) is None):
        return None
    lat = getattr(result, "latitude", None)
    lon = getattr(result, "longitude", None)
    if lat is None or lon is None or (isinstance(lat, float) and math.isnan(lat)):
        return None
    return (float(lat), float(lon))


def _trial_min_distance_miles(trial: dict, patient_lat: float, patient_lon: float) -> float:
    """Return minimum distance (miles) from patient to any of the trial's recruiting sites."""
    min_dist = float("inf")
    for col, coords_key in LOCATION_ACTIVE_COLUMNS:
        val = _safe_val(trial.get(col, "")).upper()
        if val in ("TRUE", "YES") and coords_key in TRIAL_SITE_COORDS:
            lat, lon = TRIAL_SITE_COORDS[coords_key]
            min_dist = min(min_dist, _haversine_miles(patient_lat, patient_lon, lat, lon))
    return min_dist if min_dist != float("inf") else 999.0


def _distance_bracket_label(miles: float) -> str:
    """Return bracket label for display, e.g. '1-25 miles'."""
    for lo, hi in DISTANCE_BRACKETS:
        if lo <= miles <= hi:
            if hi >= 9999:
                return "100+ miles"
            return f"{lo}-{hi} miles"
    return "100+ miles"


def _distance_bracket_sort_key(miles: float) -> int:
    """Return sort order (lower = closer)."""
    for i, (lo, hi) in enumerate(DISTANCE_BRACKETS):
        if lo <= miles <= hi:
            return i
    return len(DISTANCE_BRACKETS)


def parse_criteria_items(text: str) -> list[str]:
    """
    Parse inclusion/exclusion criteria text into discrete items.
    Splits on bullet points (•, *, -) and numbered lists.
    Returns a list of trimmed, non-empty criterion strings.
    """
    if not text or not str(text).strip():
        return []
    out = []
    for line in str(text).splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove bullet markers: •, *, -
        line = re.sub(r"^\s*[•\*\-]\s*", "", line).strip()
        # Remove numbered prefixes: 1. 2) 1) etc.
        line = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
        line = line.rstrip(";").strip()
        if line and len(line) > 3:
            out.append(line)
    seen = set()
    deduped = []
    for item in out:
        norm = item.lower()[:100]
        if norm not in seen:
            seen.add(norm)
            deduped.append(item)
    return deduped[:8]  # Cap to stay under token limit


def trial_to_context(row: dict, distance_miles: float | None = None) -> str:
    def trunc(s, n=MAX_CRITERIA_LEN):
        s = str(s) if s else ""
        return (s[:n] + "...") if len(s) > n else s

    inc_raw = row.get("inclusion_criteria", "") or ""
    exc_raw = row.get("exclusion_criteria", "") or ""
    inc_items = parse_criteria_items(inc_raw)
    exc_items = parse_criteria_items(exc_raw)

    inc_list = "\n".join(f"  - {item[:120]}" for item in inc_items[:5]) if inc_items else trunc(inc_raw, 300)
    exc_list = "\n".join(f"  - {item[:120]}" for item in exc_items[:5]) if exc_items else trunc(exc_raw, 300)

    loc_status = []
    for coords_key, label in LOCATION_LABELS:
        col = next((c for c, k in LOCATION_ACTIVE_COLUMNS if k == coords_key), None)
        if not col:
            continue
        val = _safe_val(row.get(col, ""))
        if val and str(val).upper() in ("TRUE", "YES"):
            loc_status.append(label)
    recruiting_at = ", ".join(loc_status) if loc_status else "—"
    dist_line = f"\n- **Distance from patient:** {distance_miles:.0f} miles ({_distance_bracket_label(distance_miles)})" if distance_miles is not None else ""

    return f"""## {row.get('nct_id', '')} — {row.get('title', '')}
- **Conditions:** {row.get('conditions', '')}
- **Locations:** {trunc(row.get('locations', ''), 200)}
- **Recruiting at:** {recruiting_at}{dist_line}
- **Contacts:** {trunc(row.get('contacts', ''), 150)}
- **Phase:** {row.get('phase', '')} | **Enrollment:** {row.get('enrollment', '')} | **Sponsor:** {row.get('sponsor', '')}
- **Sex:** {row.get('sex', '')} | **Age:** {row.get('minimum_age', '')}–{row.get('maximum_age', '')}
- **Brief summary:** {trunc(row.get('brief_summary', ''), 280)}
- **Interventions:** {trunc(row.get('interventions', ''), 150)}
- **Inclusion criteria (parsed for screening):**
{inc_list}
- **Exclusion criteria (parsed for screening):**
{exc_list}
"""


SYSTEM_PROMPT = """You are a clinical trial assistant for physicians. You help them find the best clinical trials in Gainesville FL, Los Angeles CA, Dallas TX, Minneapolis MN, New York NY, and Seattle WA through a guided conversation.

**CRITICAL BUSINESS RULES:**
1. **Max 3 trials:** Show at most 3 trials at any time. If more than 3 matches exist, show the top 3 and ask for additional clarification to narrow down.
2. **Patient zip required:** Always ask for the patient's zip code to assess distance. Distance brackets (priority order): 1–25 miles, 26–50 miles, 51–100 miles, 100+ miles. Sort and present trials by these brackets.
3. **85% confidence:** Only recommend trials when at least 85% confident the patient matches inclusion/exclusion criteria. Otherwise ask clarifying questions.
4. **Format when recommending:** Group by distance bracket. Example: "1–25 miles: NCT123... | 26–50 miles: NCT456..., NCT789..."

**Your approach:**
1. Ask: condition, age, sex, and patient zip code.
2. Use inclusion/exclusion to ask 1–2 qualifying questions before recommending.
3. Prioritize by distance bracket (1–25 first, then 26–50, 51–100, 100+).
4. If >3 matches: show top 3, ask for clarification.
5. If <85% confident: ask questions; do NOT recommend yet.
6. Be concise. No medical advice.
7. **No reasoning for non-matches:** Do not explain why trials are not a match (e.g., "NCT123 is not a match since..."). Only list the recommended trials. Omit any trial that does not fit—do not mention it.

**Output (at end of response):**
CONFIDENCE: <0-100>
RECOMMENDED_IDS: NCTid1,NCTid2,NCTid3 (max 3, distance order; empty if asking)"""


def _parse_recommended_ids(reply: str, all_nct_ids: set[str]) -> list[str]:
    """Extract RECOMMENDED_IDS line and return valid NCT IDs in order (max 3)."""
    m = re.search(r"RECOMMENDED_IDS:\s*([\w\s,]+)", reply, re.IGNORECASE)
    if not m:
        return []
    raw = m.group(1).strip()
    ids = [s.strip().upper() for s in re.split(r"[\s,]+", raw) if s.strip()]
    ids = [id_ for id_ in ids if id_ in all_nct_ids][:MAX_TRIALS_DISPLAY]
    return ids


def _build_conversation_messages(
    messages: list[dict],
    trials: list[dict],
    patient_zip: str | None = None,
) -> list[dict]:
    """Build full message list for the LLM, with trial context injected before the last user turn."""
    # Compute distances if we have patient zip
    patient_coords = _zip_to_coords(patient_zip) if patient_zip else None
    trial_contexts = []
    for t in trials:
        dist = None
        if patient_coords:
            dist = _trial_min_distance_miles(t, patient_coords[0], patient_coords[1])
        trial_contexts.append(trial_to_context(t, distance_miles=dist))
    context = "\n\n".join(trial_contexts)
    zip_note = f"\n**Patient zip:** {patient_zip} (used for distance calculation)" if patient_zip else ""
    injected = f"""**Relevant trials (from search):**{zip_note}

{context}

---

**Current conversation summary / latest input:**"""
    # Keep only last 6 messages to stay under token limit
    recent = messages[-6:] if len(messages) > 6 else messages
    out = [{"role": "system", "content": SYSTEM_PROMPT}]
    for i, m in enumerate(recent):
        role, content = m.get("role"), m.get("content", "")
        is_last = i == len(recent) - 1
        if role == "user" and is_last:
            out.append({"role": "user", "content": f"{injected}\n\n{content}"})
        elif role in ("user", "assistant"):
            out.append({"role": role, "content": content})
    return out


def _build_trial_preview(trial: dict, distance_miles: float | None) -> dict:
    """Add preview fields: criteria_overview, distance_miles, distance_label."""
    inc_items = parse_criteria_items(trial.get("inclusion_criteria", "") or "")
    exc_items = parse_criteria_items(trial.get("exclusion_criteria", "") or "")
    age = f"{trial.get('minimum_age', '') or '?'}–{trial.get('maximum_age', '') or '?'}"
    sex = trial.get("sex", "") or "All"
    overview_parts = [f"Age: {age}", f"Sex: {sex}"]
    if inc_items:
        overview_parts.append("Key inclusion: " + inc_items[0][:80] + ("…" if len(inc_items[0]) > 80 else ""))
    if exc_items:
        overview_parts.append("Key exclusion: " + exc_items[0][:80] + ("…" if len(exc_items[0]) > 80 else ""))
    out = dict(trial)
    out["criteria_overview"] = " | ".join(overview_parts)
    out["distance_miles"] = round(distance_miles, 0) if distance_miles is not None else None
    out["distance_label"] = _distance_bracket_label(distance_miles) if distance_miles is not None else None
    return out


def chat(messages: list[dict], patient_zip: str | None = None) -> tuple[str, list[dict]]:
    if not messages:
        return (
            "I'll help you find clinical trials for your patient. "
            "What's the patient's **primary condition**? "
            "Also share age, sex, **zip code**, and any key history.",
            [],
        )
    user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
    query_for_search = " ".join(user_msgs) if user_msgs else "clinical trials"
    trials = search_trials(query_for_search)

    # Extract zip from messages if not provided (look for 5-digit zip)
    zip_val = patient_zip
    if not zip_val:
        for m in user_msgs:
            match = re.search(r"\b(\d{5})\b", str(m))
            if match:
                zip_val = match.group(1)
                break

    # Compute distances and sort by bracket when we have zip
    patient_coords = _zip_to_coords(zip_val) if zip_val else None
    if patient_coords:
        for t in trials:
            t["_dist"] = _trial_min_distance_miles(t, patient_coords[0], patient_coords[1])
        trials = sorted(trials, key=lambda x: (_distance_bracket_sort_key(x["_dist"]), x["_dist"]))

    api_messages = _build_conversation_messages(messages, trials, patient_zip=zip_val)

    client = OpenAI()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=api_messages,
        temperature=0.3,
    )
    reply = resp.choices[0].message.content
    all_nct = {str(t.get("nct_id", "")).upper() for t in trials}
    recommended = _parse_recommended_ids(reply, all_nct)

    # Strip CONFIDENCE and RECOMMENDED_IDS from visible reply
    reply = re.sub(r"\n*\s*CONFIDENCE:\s*\d+\s*", "", reply, flags=re.IGNORECASE)
    reply = re.sub(r"\n*\s*RECOMMENDED_IDS:\s*[\w\s,]*\s*$", "", reply, flags=re.IGNORECASE).strip()

    if recommended:
        trials = [t for t in trials if str(t.get("nct_id", "")).upper() in recommended]
        trials = sorted(
            trials,
            key=lambda t: recommended.index(str(t.get("nct_id", "")).upper())
            if str(t.get("nct_id", "")).upper() in recommended
            else 999,
        )[:MAX_TRIALS_DISPLAY]
        # Build preview with distance
        out_trials = []
        for t in trials:
            dist = t.pop("_dist", None)
            out_trials.append(_build_trial_preview(t, dist))
        trials = out_trials
    else:
        trials = []
    return reply, trials


def draft_email(nct_id: str, patient_summary: str) -> str:
    row_dict = get_trial_by_id(nct_id)
    if row_dict is None:
        raise HTTPException(404, f"Trial {nct_id} not found")

    contacts = row_dict.get("contacts", "")
    title = row_dict.get("title", "")
    conditions = row_dict.get("conditions", "")

    prompt = f"""You are helping a physician draft a professional referral email to a clinical trial coordinator.

**Trial:** {title}
**NCT ID:** {nct_id}
**Conditions:** {conditions}
**Study contact(s):** {contacts}

**Physician/patient info (from the physician):**
{patient_summary}

Write a concise, professional email that:
1. Introduces the physician and states the purpose (referring a potential participant)
2. Uses the physician's name and occupation from the info above when provided
3. Briefly summarizes the patient's relevant history and why they may be a good fit (if any)
4. Asks to discuss next steps (screening, enrollment)
5. Is respectful of the coordinator's time and suitable for a medical professional

Use a professional tone. Include a suggested subject line at the very top, then a blank line, then the email body.
Format as plain text suitable for copy-paste into an email client."""

    client = OpenAI()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_loaded()
    yield
    # cleanup if needed


app = FastAPI(title="Trial Finder API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:5174", "http://127.0.0.1:5174",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    patient_zip: str | None = None


class ChatResponse(BaseModel):
    reply: str
    trials: list[dict]


class DraftEmailRequest(BaseModel):
    nct_id: str
    patient_summary: str


class DraftEmailResponse(BaseModel):
    email: str


def get_trial_by_id(nct_id: str) -> dict | None:
    if _rows is None:
        ensure_loaded()
    nct = str(nct_id).strip().upper()
    for r in _rows or []:
        if str(r.get("nct_id", "")).strip().upper() == nct:
            return dict(r)
    return None


@app.get("/api/health")
def health():
    ensure_loaded()
    return {"status": "ok", "trials_loaded": len(_rows) if _rows is not None else 0}


@app.get("/api/trial/{nct_id}")
def get_trial(nct_id: str):
    ensure_loaded()
    trial = get_trial_by_id(nct_id)
    if trial is None:
        raise HTTPException(404, f"Trial {nct_id} not found")
    return trial


@app.post("/api/chat", response_model=ChatResponse)
def post_chat(req: ChatRequest):
    ensure_loaded()
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(503, "OPENAI_API_KEY not configured")
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    reply, trials = chat(msgs, patient_zip=req.patient_zip)
    return ChatResponse(reply=reply, trials=trials)


@app.post("/api/draft-email", response_model=DraftEmailResponse)
def post_draft_email(req: DraftEmailRequest):
    ensure_loaded()
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(503, "OPENAI_API_KEY not configured")
    email = draft_email(req.nct_id, req.patient_summary)
    return DraftEmailResponse(email=email)
