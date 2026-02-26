# Clinical Trial Navigator

A web app for physicians to find clinical trials in Gainesville FL, Los Angeles CA, Dallas TX, Minneapolis MN, New York NY, and Seattle WA, with AI-assisted matching and referral email drafting.

## Features

- **Multi-location search** — Gainesville, Los Angeles, Dallas, Minneapolis, New York, Seattle
- **Distance-based matching** — 1–25, 26–50, 51–100, 100+ mile brackets from patient zip
- **Criteria-based screening** — AI asks qualifying questions from inclusion/exclusion criteria
- **Top 3 trials** — Max 3 recommendations, sorted by proximity
- **Draft referral emails** — One-click draft for trial coordinators

## Architecture

- **Backend**: FastAPI (Python) — trial search (TF-IDF), OpenAI chat, email drafting
- **Frontend**: React + Tailwind (package manager: Yarn)

## Setup

1. **Install backend dependencies**
   ```bash
   cd backend && pip install -r requirements.txt
   ```

2. **Install frontend dependencies**
   ```bash
   cd frontend && yarn install
   ```
   *(No global Yarn? Use `npx yarn install` then `npx yarn dev`.)*

3. **Environment**
   Create a `.env` file in the **project root** (backend loads it when run from `backend/`):
   ```
   OPENAI_API_KEY=sk-your-key
   ```
   Get a key at [platform.openai.com](https://platform.openai.com/api-keys).

4. **Trial data**
   Ensure trial data exists (from project root):
   ```bash
   cd backend && python a.py          # Gainesville + Los Angeles
   # or
   cd backend && python a.py --gainesville-only   # Gainesville only
   ```

## Run

1. **Start the API** (terminal 1)
   ```bash
   cd backend && python -m uvicorn api.main:app --reload --port 8000
   ```

2. **Start the frontend** (terminal 2)
   ```bash
   cd frontend && yarn dev
   ```

3. Open [http://localhost:5173](http://localhost:5173)

## Project structure

- `backend/` — FastAPI API (`api/main.py`), trial fetch script (`a.py`), requirements, trial CSVs
- `frontend/` — React app (Yarn only; use `yarn install` and `yarn dev`—avoid adding npm or duplicate tooling).
