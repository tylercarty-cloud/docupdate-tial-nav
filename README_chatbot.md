# Trial Finder — React + API

A professional web app for physicians to find Gainesville clinical trials and draft referral emails to study contacts.

## Architecture

- **Backend**: FastAPI (Python) — trial search, Groq chat, email drafting
- **Frontend**: React + Tailwind (Yarn)

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
   Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=gsk_your-key
   ```
   Get a key at [console.groq.com](https://console.groq.com).

4. **Trial data**
   Ensure trial data exists (run `cd backend && python a.py --all-conditions` if needed).

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

## Features

- Semantic search over trial data
- AI-assisted trial matching for patient descriptions
- Draft referral emails to study contacts
- Copy-to-clipboard for email drafts
