"""
Fetch clinical trials from ClinicalTrials.gov for multiple locations.

Locations: Gainesville FL, Los Angeles CA, Dallas TX, Minneapolis MN, New York NY, Seattle WA.

Modes:
- Default: all conditions, all locations → active_trials.csv
- --gainesville-only: Gainesville only (legacy) → gainesville_active_trials.csv
- --heart-attack: heart-attack only, Gainesville → gainesville_heart_attack_trials.csv

CSV columns (always in this order):
  nct_id, title, overall_status, locations, location_sites,
  gainesville_*, los_angeles_*, dallas_*, minneapolis_*, new_york_*, seattle_* (recruitment_status, active_recruiting each),
  conditions, contacts, inclusion_criteria, exclusion_criteria,
  phase, enrollment, enrollment_type, sponsor, study_type,
  sex, minimum_age, maximum_age, healthy_volunteers,
  brief_summary, interventions, primary_outcomes, secondary_outcomes,
  start_date, primary_completion_date, completion_date, status_verified_date
"""
import csv
import re
import requests

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

CSV_COLUMNS = [
    "nct_id",
    "title",
    "overall_status",
    "locations",
    "location_sites",
    "gainesville_recruitment_status",
    "gainesville_active_recruiting",
    "los_angeles_recruitment_status",
    "los_angeles_active_recruiting",
    "dallas_recruitment_status",
    "dallas_active_recruiting",
    "minneapolis_recruitment_status",
    "minneapolis_active_recruiting",
    "new_york_recruitment_status",
    "new_york_active_recruiting",
    "seattle_recruitment_status",
    "seattle_active_recruiting",
    "conditions",
    "contacts",
    "inclusion_criteria",
    "exclusion_criteria",
    "phase",
    "enrollment",
    "enrollment_type",
    "sponsor",
    "study_type",
    "sex",
    "minimum_age",
    "maximum_age",
    "healthy_volunteers",
    "brief_summary",
    "interventions",
    "primary_outcomes",
    "secondary_outcomes",
    "start_date",
    "primary_completion_date",
    "completion_date",
    "status_verified_date",
]

# Locations we fetch: (query.locn string, city match, state matches)
LOCATIONS_CONFIG = [
    ("Gainesville Florida", "GAINESVILLE", ("FLORIDA", "FL")),
    ("Los Angeles California", "LOS ANGELES", ("CALIFORNIA", "CA")),
    ("Dallas Texas", "DALLAS", ("TEXAS", "TX")),
    ("Minneapolis Minnesota", "MINNEAPOLIS", ("MINNESOTA", "MN")),
    ("New York New York", "NEW YORK", ("NEW YORK", "NY")),
    ("Seattle Washington", "SEATTLE", ("WASHINGTON", "WA")),
]

# Map query_locn substring to CSV column prefix (e.g. "Gainesville" -> "gainesville")
LOCATION_COLUMN_PREFIX = {
    "Gainesville": "gainesville",
    "Los Angeles": "los_angeles",
    "Dallas": "dallas",
    "Minneapolis": "minneapolis",
    "New York": "new_york",
    "Seattle": "seattle",
}

FIELDS = [
    "protocolSection.identificationModule.nctId",
    "protocolSection.identificationModule.briefTitle",
    "protocolSection.statusModule.overallStatus",
    "protocolSection.statusModule.statusVerifiedDate",
    "protocolSection.statusModule.startDateStruct",
    "protocolSection.statusModule.primaryCompletionDateStruct",
    "protocolSection.statusModule.completionDateStruct",
    "protocolSection.conditionsModule.conditions",
    "protocolSection.eligibilityModule.eligibilityCriteria",
    "protocolSection.eligibilityModule.sex",
    "protocolSection.eligibilityModule.minimumAge",
    "protocolSection.eligibilityModule.maximumAge",
    "protocolSection.eligibilityModule.healthyVolunteers",
    "protocolSection.contactsLocationsModule.locations",
    "protocolSection.contactsLocationsModule.centralContacts",
    "protocolSection.contactsLocationsModule.overallOfficials",
    "protocolSection.designModule.studyType",
    "protocolSection.designModule.phases",
    "protocolSection.designModule.enrollmentInfo",
    "protocolSection.designModule.designInfo",
    "protocolSection.armsInterventionsModule",
    "protocolSection.outcomesModule.primaryOutcomes",
    "protocolSection.descriptionModule.briefSummary",
    "protocolSection.sponsorCollaboratorsModule.leadSponsor",
]

# Match start of exclusion section (must be after newline to avoid mid-sentence).
# Order matters: longer / more specific first. Use \s* so "Key exclusion:" (no space before :) matches.
# Also match "* Key exclusion:" or "- Key exclusion:" when it appears as a bullet line.
EXCLUSION_SPLIT = re.compile(
    r"\n\s*Key exclusion criteria include but are not limited to\s*:"
    r"|\n\s*Key exclusion criteria\s*:"
    r"|\n\s*Key exclusion\s*:"
    r"|\n\s*[\*\-]\s*Key exclusion\s*:"
    r"|\n\s*(?:Key\s+)?Exclusion\s*(?:Criteria\s*)?\s*:"
    r"|\n\s*Exclusion\s*:"
    r"|\n\s*(?:Key\s+)?Exclusion\s*(?:Criteria\s*)?\s*\n"
    r"|\n\s*Exclusion\s+criteria\s*:",
    re.IGNORECASE,
)

INCLUSION_HEADERS = re.compile(
    r"^(?:Key\s+)?Inclusion\s*(?:Criteria\s*)?\s*:"
    r"|^Inclusion\s*:"
    r"|^Key inclusion criteria include but are not limited to\s*:",
    re.IGNORECASE | re.MULTILINE,
)
EXCLUSION_HEADERS = re.compile(
    r"^(?:Key\s+)?Exclusion\s*(?:Criteria\s*)?\s*:"
    r"|^Exclusion\s*:"
    r"|^Key exclusion criteria include but are not limited to\s*:"
    r"|^Key exclusion criteria\s*:"
    r"|^Key exclusion\s*:"
    r"|^\s*[\*\-]\s*Key exclusion\s*:",
    re.IGNORECASE | re.MULTILINE,
)
# Leftover ": " or ":\n" at start of block after split.
LEADING_COLON = re.compile(r"^[\s:]+")

BULLET = re.compile(r"^\s*[\*\-]\s*|\s*[\*\-]\s*$")
NUMBERED = re.compile(r"^\s*\d+[\.\)]\s*")


def _normalize_criteria_block(
    raw: str,
    *,
    strip_inclusion_header: bool = True,
    strip_exclusion_header: bool = True,
) -> str:
    """Normalize a criteria block to consistent '• item' format."""
    if not raw or not raw.strip():
        return ""
    text = raw.strip()
    text = LEADING_COLON.sub("", text).strip()
    if strip_inclusion_header:
        text = INCLUSION_HEADERS.sub("", text)
    if strip_exclusion_header:
        text = EXCLUSION_HEADERS.sub("", text)
    text = text.strip()
    text = LEADING_COLON.sub("", text).strip()
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = BULLET.sub("", line).strip()
        line = NUMBERED.sub("", line).strip()
        if not line:
            continue
        lines.append(f"• {line}")
    return "\n".join(lines) if lines else ""


def parse_eligibility(text: str) -> tuple[str, str]:
    """
    Split eligibility into inclusion and exclusion, normalize both.
    Always returns (inclusion, exclusion) as separate strings; never combined.
    """
    text = (text or "").strip()
    if not text:
        return "", ""

    match = EXCLUSION_SPLIT.search(text)
    if match:
        inc_raw = text[: match.start()].strip()
        exc_raw = text[match.start() :].strip()
    else:
        inc_raw = text
        exc_raw = ""

    inc_raw = LEADING_COLON.sub("", inc_raw).strip()
    exc_raw = LEADING_COLON.sub("", exc_raw).strip()

    inclusion = _normalize_criteria_block(
        inc_raw, strip_inclusion_header=True, strip_exclusion_header=False
    )
    exclusion = _normalize_criteria_block(
        exc_raw, strip_inclusion_header=False, strip_exclusion_header=True
    )
    return inclusion, exclusion


def _format_date(d: dict | None) -> str:
    if not d or not isinstance(d, dict):
        return ""
    return d.get("date") or ""


def _format_interventions(arms_module: dict | list | None) -> str:
    if not arms_module or isinstance(arms_module, list):
        return ""
    interventions = arms_module.get("interventions") or []
    names = [i["name"] for i in interventions if isinstance(i, dict) and i.get("name")]
    return "; ".join(names)


def _format_primary_outcomes(outcomes: list | None) -> str:
    if not outcomes:
        return ""
    parts = []
    for o in outcomes:
        if isinstance(o, dict):
            m = o.get("measure") or ""
            d = o.get("description") or ""
            tf = o.get("timeFrame") or ""
            if m or d or tf:
                parts.append(f"{m}: {d} ({tf})".strip(" :"))
    return " | ".join(parts)


def _fetch_trials_for_location(
    query_locn: str,
    city_match: str,
    state_matches: tuple[str, ...],
    *,
    recruiting_only: bool = True,
    all_conditions: bool = True,
) -> list[dict]:
    """Fetch trials with at least one site in the given city/state."""
    trials = []
    next_token = None
    params_base = {
        "pageSize": 100,
        "query.locn": query_locn,
        "filter.overallStatus": "RECRUITING,NOT_YET_RECRUITING",
        "fields": ",".join(FIELDS),
        "format": "json",
    }
    if not all_conditions:
        params_base["query.cond"] = (
            "heart attack OR myocardial infarction OR MI OR acute coronary syndrome"
        )

    while True:
        params = dict(params_base)
        if next_token:
            params["pageToken"] = next_token

        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        studies = data.get("studies", [])
        if not studies:
            break

        for s in studies:
            section = s.get("protocolSection", {})
            overall_status = section.get("statusModule", {}).get("overallStatus", "")
            if overall_status not in ("RECRUITING", "NOT_YET_RECRUITING"):
                continue

            locations = section.get("contactsLocationsModule", {}).get("locations", [])
            matching_locs = [
                loc
                for loc in locations
                if (loc.get("city") or "").upper() == city_match
                and (loc.get("state") or "").upper() in state_matches
            ]
            if not matching_locs:
                continue

            statuses = []
            active = False
            for loc in matching_locs:
                st = (loc.get("status") or "").strip().upper()
                fac = loc.get("facility") or "Unknown"
                statuses.append(f"{fac} ({st})" if st else fac)
                if st in ("RECRUITING", "NOT_YET_RECRUITING"):
                    active = True

            if recruiting_only and not active:
                continue

            locs_out = []
            contacts_out = []
            clm = section.get("contactsLocationsModule", {})

            for c in clm.get("centralContacts", []):
                contacts_out.append(
                    f"{c.get('name', '')} | {c.get('phone', '')} | {c.get('email', '')}"
                )
            for o in clm.get("overallOfficials", []):
                contacts_out.append(
                    f"{o.get('name', '')} | {o.get('phone', '')} | {o.get('email', '')}"
                )

            for loc in matching_locs:
                city = (loc.get("city") or "").title()
                state = (loc.get("state") or "").upper()
                fac = loc.get("facility") or ""
                country = loc.get("country") or ""
                locs_out.append(f"{fac}, {city}, {state}, {country}")
                for c in loc.get("contacts") or []:
                    contacts_out.append(
                        f"{c.get('name', '')} | {c.get('phone', '')} | {c.get('email', '')}"
                    )

            if not locs_out:
                continue

            conditions = section.get("conditionsModule", {}).get("conditions", []) or []
            if not all_conditions:
                cond_text = " ".join(conditions).lower()
                title = (
                    section.get("identificationModule", {}) or {}
                ).get("briefTitle") or ""
                title = title.lower()
                keywords = [
                    "heart attack",
                    "myocardial infarction",
                    "acute coronary",
                    "coronary syndrome",
                    "stemi",
                    "nstemi",
                    "mi ",
                    " mi",
                ]
                if not any(k in cond_text or k in title for k in keywords):
                    continue

            eligibility_text = (
                section.get("eligibilityModule", {}) or {}
            ).get("eligibilityCriteria") or ""
            inclusion, exclusion = parse_eligibility(eligibility_text)

            seen = set()
            unique_contacts = []
            for c in contacts_out:
                if c not in seen:
                    seen.add(c)
                    unique_contacts.append(c)

            status_mod = section.get("statusModule", {}) or {}
            design = section.get("designModule", {}) or {}
            enroll = design.get("enrollmentInfo") or {}
            elig = section.get("eligibilityModule", {}) or {}
            arms = section.get("armsInterventionsModule") or {}
            outcomes_mod = section.get("outcomesModule") or {}
            primary_outcomes = outcomes_mod.get("primaryOutcomes")
            secondary_outcomes = outcomes_mod.get("secondaryOutcomes")
            desc = (section.get("descriptionModule") or {}).get("briefSummary") or ""

            hv = elig.get("healthyVolunteers")
            if hv is True:
                healthy_volunteers = "YES"
            elif hv is False:
                healthy_volunteers = "NO"
            else:
                healthy_volunteers = ""

            nct_id = (section.get("identificationModule", {}) or {}).get("nctId", "")
            row = {
                "nct_id": nct_id,
                "title": (section.get("identificationModule", {}) or {}).get(
                    "briefTitle", ""
                ),
                "overall_status": overall_status,
                "locations": "; ".join(locs_out),
                "location_sites": query_locn,
                "gainesville_recruitment_status": "",
                "gainesville_active_recruiting": False,
                "los_angeles_recruitment_status": "",
                "los_angeles_active_recruiting": False,
                "dallas_recruitment_status": "",
                "dallas_active_recruiting": False,
                "minneapolis_recruitment_status": "",
                "minneapolis_active_recruiting": False,
                "new_york_recruitment_status": "",
                "new_york_active_recruiting": False,
                "seattle_recruitment_status": "",
                "seattle_active_recruiting": False,
                "conditions": "; ".join(conditions),
                "contacts": "; ".join(unique_contacts),
                "inclusion_criteria": inclusion,
                "exclusion_criteria": exclusion,
                "phase": "; ".join(design.get("phases") or []),
                "enrollment": enroll.get("count") or "",
                "enrollment_type": enroll.get("type") or "",
                "sponsor": (
                    (section.get("sponsorCollaboratorsModule", {}) or {})
                    .get("leadSponsor")
                    or {}
                ).get("name", ""),
                "study_type": design.get("studyType") or "",
                "sex": elig.get("sex") or "",
                "minimum_age": elig.get("minimumAge") or "",
                "maximum_age": elig.get("maximumAge") or "",
                "healthy_volunteers": healthy_volunteers,
                "brief_summary": (desc[:2000] + "…") if len(desc) > 2000 else desc,
                "interventions": _format_interventions(arms),
                "primary_outcomes": _format_primary_outcomes(primary_outcomes),
                "secondary_outcomes": _format_primary_outcomes(secondary_outcomes),
                "start_date": _format_date(status_mod.get("startDateStruct")),
                "primary_completion_date": _format_date(
                    status_mod.get("primaryCompletionDateStruct")
                ),
                "completion_date": _format_date(
                    status_mod.get("completionDateStruct")
                ),
                "status_verified_date": status_mod.get("statusVerifiedDate") or "",
            }
            # Set location-specific fields based on which city we're fetching
            for key_substr, prefix in LOCATION_COLUMN_PREFIX.items():
                if key_substr in query_locn:
                    row[f"{prefix}_recruitment_status"] = "; ".join(statuses)
                    row[f"{prefix}_active_recruiting"] = active
                    break
            trials.append(row)

        next_token = data.get("nextPageToken")
        if not next_token:
            break

    return trials


def fetch_multi_location_trials(
    *,
    recruiting_only: bool = True,
    all_conditions: bool = True,
    locations: list[tuple[str, str, tuple[str, ...]]] | None = None,
) -> list[dict]:
    """
    Fetch trials from multiple locations and merge by NCT ID.
    Trials appearing in both locations get combined location fields.
    """
    locs = locations or LOCATIONS_CONFIG
    by_nct: dict[str, dict] = {}

    for query_locn, city_match, state_matches in locs:
        print(f"Fetching {query_locn}...", flush=True)
        rows = _fetch_trials_for_location(
            query_locn,
            city_match,
            state_matches,
            recruiting_only=recruiting_only,
            all_conditions=all_conditions,
        )
        print(f"  Found {len(rows)} trials", flush=True)
        for row in rows:
            nct = row.get("nct_id", "")
            if nct in by_nct:
                existing = by_nct[nct]
                existing["locations"] = existing["locations"] + " | " + row["locations"]
                loc_sites = existing.get("location_sites", "")
                if row["location_sites"] and row["location_sites"] not in loc_sites:
                    existing["location_sites"] = (
                        f"{loc_sites}; {row['location_sites']}" if loc_sites else row["location_sites"]
                    )
                for prefix in LOCATION_COLUMN_PREFIX.values():
                    status_col = f"{prefix}_recruitment_status"
                    active_col = f"{prefix}_active_recruiting"
                    if row.get(status_col):
                        existing[status_col] = row[status_col]
                        existing[active_col] = row.get(active_col, False)
            else:
                by_nct[nct] = dict(row)

    return list(by_nct.values())


def fetch_gainesville_trials(
    *,
    gainesville_recruiting_only: bool = True,
    all_conditions: bool = False,
) -> list[dict]:
    """Legacy: Fetch Gainesville-only trials (for backward compatibility)."""
    return _fetch_trials_for_location(
        "Gainesville Florida",
        "GAINESVILLE",
        ("FLORIDA", "FL"),
        recruiting_only=gainesville_recruiting_only,
        all_conditions=all_conditions,
    )


def write_trials_csv(trials: list[dict], path: str) -> None:
    """Write trials to CSV with all columns in fixed order."""
    bool_cols = tuple(f"{p}_active_recruiting" for p in LOCATION_COLUMN_PREFIX.values())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore",
            quoting=csv.QUOTE_MINIMAL,
        )
        w.writeheader()
        for r in trials:
            out = {k: r.get(k, "") for k in CSV_COLUMNS}
            for col in bool_cols:
                if isinstance(out.get(col), bool):
                    out[col] = "TRUE" if out[col] else "FALSE"
            w.writerow(out)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Fetch clinical trials from ClinicalTrials.gov (Gainesville, Los Angeles, Dallas, Minneapolis, New York, Seattle)."
    )
    p.add_argument(
        "--all-sites",
        action="store_true",
        help="Include trials with any site (even if not recruiting there).",
    )
    p.add_argument(
        "--gainesville-only",
        action="store_true",
        help="Fetch Gainesville, FL only (output: gainesville_active_trials.csv).",
    )
    p.add_argument(
        "--heart-attack",
        action="store_true",
        help="Fetch heart-attack related trials, Gainesville only (output: gainesville_heart_attack_trials.csv).",
    )
    args = p.parse_args()

    if args.heart_attack:
        trials = fetch_gainesville_trials(
            gainesville_recruiting_only=not args.all_sites,
            all_conditions=False,
        )
        out_path = "gainesville_heart_attack_trials.csv"
        label = "heart-attack (Gainesville only)"
    elif args.gainesville_only:
        trials = fetch_gainesville_trials(
            gainesville_recruiting_only=not args.all_sites,
            all_conditions=True,
        )
        out_path = "gainesville_active_trials.csv"
        label = "all conditions (Gainesville only)"
    else:
        trials = fetch_multi_location_trials(
            recruiting_only=not args.all_sites,
            all_conditions=True,
        )
        out_path = "active_trials.csv"
        label = "all conditions (all 6 locations)"

    write_trials_csv(trials, out_path)
    mode = "all sites" if args.all_sites else "recruiting only"
    print(f"Saved {len(trials)} trials ({label}) to {out_path} ({mode})")
