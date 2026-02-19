# C:\MOCK\app.py
import os
import json
import datetime as dt
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI

from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# -----------------------------
# DEV ONLY: allow OAuth over HTTP for localhost
# -----------------------------
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

# -----------------------------
# Paths / Config
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.readonly",
]

GOOGLE_CLIENT_SECRET_FILE = Path(
    os.getenv("GOOGLE_CLIENT_SECRET_FILE", str(BASE_DIR / "google_client_secret.json"))
)
TOKEN_FILE = Path(os.getenv("GOOGLE_TOKEN_FILE", str(BASE_DIR / "google_token.json")))

HOST = os.getenv("MOCK_HOST", "127.0.0.1")
PORT = int(os.getenv("MOCK_PORT", "8000"))
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", f"http://{HOST}:{PORT}/auth/google/callback")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")  # matches your /health output

client = OpenAI()  # reads OPENAI_API_KEY from env

app = FastAPI()

# Allow your static HTML to call this API locally
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "ok": True,
        "mode": "mvp",
        "google_secret_found": GOOGLE_CLIENT_SECRET_FILE.exists(),
        "token_found": TOKEN_FILE.exists(),
        "redirect_uri": REDIRECT_URI,
        "model": OPENAI_MODEL,
    }


# -----------------------------
# Google auth helpers
# -----------------------------
def load_creds() -> Optional[Credentials]:
    if TOKEN_FILE.exists():
        return Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    return None


def save_creds(creds: Credentials):
    TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")


def require_google_secret():
    if not GOOGLE_CLIENT_SECRET_FILE.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                f"Missing Google OAuth file: {GOOGLE_CLIENT_SECRET_FILE}. "
                f"Place google_client_secret.json in C:\\MOCK or set env GOOGLE_CLIENT_SECRET_FILE."
            ),
        )


def calendar_service():
    creds = load_creds()
    if not creds:
        raise HTTPException(status_code=401, detail="Not authenticated. Visit /auth/google first.")
    return build("calendar", "v3", credentials=creds)


@app.get("/auth/google")
def auth_google():
    require_google_secret()

    flow = Flow.from_client_secrets_file(
        str(GOOGLE_CLIENT_SECRET_FILE),
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
    )
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    return RedirectResponse(auth_url)


@app.get("/auth/google/callback")
def auth_google_callback(request: Request):
    require_google_secret()

    flow = Flow.from_client_secrets_file(
        str(GOOGLE_CLIENT_SECRET_FILE),
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
    )
    flow.fetch_token(authorization_response=str(request.url))
    save_creds(flow.credentials)
    return JSONResponse({"ok": True, "message": "Google Calendar connected. You can close this tab."})


# -----------------------------
# OpenAI Structured Draft (Option A: attendees are EMAIL ONLY)
# NOTE: OpenAI strict schema validator requires REQUIRED to include EVERY property key.
# So we make all fields required and allow empty strings/arrays as "optional".
# -----------------------------
DRAFT_SCHEMA = {
    "name": "InterviewInviteDraft",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string"},
            "duration_minutes": {"type": "integer", "minimum": 15, "maximum": 180},
            "attendees": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"email": {"type": "string"}},
                    "required": ["email"],
                },
            },
            "candidate": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
                "required": ["name", "email"],
            },
            "role": {"type": "string"},
            "job_description_summary": {"type": "string"},
            "talking_points": {"type": "array", "items": {"type": "string"}},
            "agenda": {"type": "array", "items": {"type": "string"}},
            "location": {"type": "string"},
            "notes_for_invite": {"type": "string"},
        },
        # IMPORTANT: include EVERY key from properties
        "required": [
            "title",
            "duration_minutes",
            "attendees",
            "candidate",
            "role",
            "job_description_summary",
            "talking_points",
            "agenda",
            "location",
            "notes_for_invite",
        ],
    },
}


def get_openai_output_text(resp) -> str:
    # Primary
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text
    # Fallback
    try:
        chunks = []
        for item in resp.output:
            for c in item.content:
                if hasattr(c, "text") and c.text:
                    chunks.append(c.text)
        return "\n".join(chunks).strip()
    except Exception:
        return ""


@app.post("/api/invite/draft")
async def invite_draft(payload: Dict[str, Any]):
    # Minimal validation
    candidate_email = (payload.get("candidate_email") or "").strip()
    candidate_name = (payload.get("candidate_name") or "").strip() or "Candidate"
    role = (payload.get("role") or "").strip()

    if not candidate_email:
        raise HTTPException(status_code=400, detail="candidate_email is required")
    if not role:
        raise HTTPException(status_code=400, detail="role is required")

    # Interviewers -> attendees (email only for schema)
    raw_interviewers = payload.get("interviewers") or []
    attendee_emails = []
    for x in raw_interviewers:
        em = (x.get("email") or "").strip()
        if em:
            attendee_emails.append(em)

    # Allow direct "attendees" too
    raw_attendees = payload.get("attendees") or []
    for x in raw_attendees:
        em = (x.get("email") or "").strip()
        if em:
            attendee_emails.append(em)

    attendee_emails = sorted(set([e.lower() for e in attendee_emails]))
    if len(attendee_emails) == 0:
        raise HTTPException(status_code=400, detail="At least one interviewer/attendee email is required")

    jd = (payload.get("job_description") or "").strip()
    ai_context = (payload.get("ai_context") or "").strip()
    location = (payload.get("location") or "Google Meet (auto)").strip()

    # Duration
    duration_minutes = payload.get("duration_minutes")
    try:
        duration_minutes = int(duration_minutes) if duration_minutes else 60
    except Exception:
        duration_minutes = 60
    duration_minutes = max(15, min(180, duration_minutes))

    # Talking points normalization
    tp = payload.get("talking_points", [])
    if isinstance(tp, str):
        tp_list = [x.strip(" -\t") for x in tp.split("\n") if x.strip()]
    elif isinstance(tp, list):
        tp_list = [str(x).strip() for x in tp if str(x).strip()]
    else:
        tp_list = []

    # Prompt: force all required fields to exist even if empty
    prompt = f"""
You are an interview scheduling assistant for DevReady.
Return ONLY strict JSON that matches the provided schema.

Inputs:
Candidate: {candidate_name} <{candidate_email}>
Attendee emails (interviewers/client): {attendee_emails}
Role: {role}
Job description: {jd}
Talking points (input): {tp_list}
Extra context from earlier chat: {ai_context}

Rules (IMPORTANT):
- Output MUST include every field in the schema, even if empty.
- title: include role + "Interview"
- duration_minutes: {duration_minutes}
- candidate: include BOTH name and email
- attendees: array of objects with ONLY {{"email":"..."}}
- job_description_summary: 3–6 lines. If no JD provided, write "N/A".
- talking_points: 3–6 items. If none provided, infer from the role and context.
- agenda: 5–8 concise items for a first-round screen.
- location: EXACTLY "{location}"
- notes_for_invite: include what to prepare + any links/credentials if present in context.
"""

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[{"role": "user", "content": prompt}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": DRAFT_SCHEMA["name"],
                    "schema": DRAFT_SCHEMA["schema"],
                    "strict": True,
                }
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")

    out_text = get_openai_output_text(resp)
    if not out_text:
        raise HTTPException(status_code=500, detail="OpenAI returned empty output_text")

    try:
        draft_json = json.loads(out_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI JSON: {e}. Raw: {out_text[:400]}")

    # Enforce required fields defensively (keep strict shape)
    draft_json.setdefault("title", f"{role} Interview")
    draft_json.setdefault("duration_minutes", duration_minutes)
    draft_json.setdefault("role", role)
    draft_json.setdefault("location", location)
    draft_json.setdefault("job_description_summary", "N/A" if not jd else draft_json.get("job_description_summary", ""))
    draft_json.setdefault("talking_points", tp_list[:6] if tp_list else [])
    draft_json.setdefault("agenda", draft_json.get("agenda", []) or [])
    draft_json.setdefault("notes_for_invite", draft_json.get("notes_for_invite", "") or "")

    # Candidate
    cand = draft_json.get("candidate") or {}
    cand.setdefault("name", candidate_name)
    cand.setdefault("email", candidate_email)
    draft_json["candidate"] = cand

    # Attendees enforce = attendee_emails
    draft_json["attendees"] = [{"email": em} for em in attendee_emails]

    # Clamp duration
    try:
        draft_json["duration_minutes"] = int(draft_json.get("duration_minutes", duration_minutes))
    except Exception:
        draft_json["duration_minutes"] = duration_minutes
    draft_json["duration_minutes"] = max(15, min(180, draft_json["duration_minutes"]))

    # Ensure talking_points is an array
    if isinstance(draft_json.get("talking_points"), str):
        draft_json["talking_points"] = [x.strip() for x in draft_json["talking_points"].split("\n") if x.strip()]
    if not isinstance(draft_json.get("talking_points"), list):
        draft_json["talking_points"] = tp_list[:6] if tp_list else []

    # Ensure agenda list
    if not isinstance(draft_json.get("agenda"), list):
        draft_json["agenda"] = []

    return draft_json


# -----------------------------
# Create Calendar Event
# -----------------------------
def overlaps(s1: dt.datetime, e1: dt.datetime, s2: dt.datetime, e2: dt.datetime) -> bool:
    return s1 < e2 and s2 < e1


@app.post("/api/invite/create")
async def invite_create(payload: Dict[str, Any]):
    svc = calendar_service()

    draft = payload.get("draft")
    if not isinstance(draft, dict):
        raise HTTPException(status_code=400, detail="draft is required and must be an object")

    tz = payload.get("timezone", "America/Denver")
    cal_id = payload.get("calendar_id", "primary")

    tw_start = payload.get("time_window_start_iso")
    tw_end = payload.get("time_window_end_iso")
    if not tw_start or not tw_end:
        raise HTTPException(status_code=400, detail="time_window_start_iso and time_window_end_iso are required")

    # Parse ISO datetimes; accept Z
    try:
        start_dt = dt.datetime.fromisoformat(tw_start.replace("Z", "+00:00"))
        end_dt = dt.datetime.fromisoformat(tw_end.replace("Z", "+00:00"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ISO datetime format for time window")

    if end_dt <= start_dt:
        raise HTTPException(status_code=400, detail="time window end must be after start")

    dur = dt.timedelta(minutes=int(draft.get("duration_minutes", 60)))

    # Freebusy
    try:
        fb = svc.freebusy().query(
            body={
                "timeMin": start_dt.isoformat(),
                "timeMax": end_dt.isoformat(),
                "timeZone": tz,
                "items": [{"id": cal_id}],
            }
        ).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Google freebusy error: {e}")

    busy = (fb.get("calendars", {}).get(cal_id, {}) or {}).get("busy", [])

    # Find first free slot in 15-min increments
    slot_start = start_dt
    found = None

    while slot_start + dur <= end_dt:
        slot_end = slot_start + dur
        conflict = False

        for b in busy:
            bs = dt.datetime.fromisoformat(b["start"].replace("Z", "+00:00"))
            be = dt.datetime.fromisoformat(b["end"].replace("Z", "+00:00"))
            if overlaps(slot_start, slot_end, bs, be):
                conflict = True
                break

        if not conflict:
            found = (slot_start, slot_end)
            break

        slot_start += dt.timedelta(minutes=15)

    if not found:
        return JSONResponse({"ok": False, "error": "No free slot found in the provided window."}, status_code=409)

    slot_start, slot_end = found

    # Attendees
    attendees = []

    cand = draft.get("candidate") or {}
    cand_email = (cand.get("email") or "").strip()
    if cand_email:
        attendees.append({"email": cand_email})

    for a in (draft.get("attendees") or []):
        em = (a.get("email") or "").strip()
        if em:
            attendees.append({"email": em})

    # De-dupe
    seen = set()
    uniq = []
    for a in attendees:
        em = a["email"].lower()
        if em not in seen:
            seen.add(em)
            uniq.append(a)
    attendees = uniq

    description = (
        f"Role: {draft.get('role','')}\n\n"
        f"JD Summary:\n{draft.get('job_description_summary','')}\n\n"
        f"Agenda:\n- " + "\n- ".join(draft.get("agenda", [])) + "\n\n"
        f"Talking Points:\n- " + "\n- ".join(draft.get("talking_points", [])) + "\n\n"
        f"Notes:\n{draft.get('notes_for_invite','')}\n"
    )

    event = {
        "summary": draft.get("title", "DevReady Interview"),
        "location": draft.get("location", "Google Meet"),
        "description": description,
        "start": {"dateTime": slot_start.isoformat(), "timeZone": tz},
        "end": {"dateTime": slot_end.isoformat(), "timeZone": tz},
        "attendees": attendees,
        "conferenceData": {
            "createRequest": {
                "requestId": f"devready-{int(dt.datetime.utcnow().timestamp())}",
                "conferenceSolutionKey": {"type": "hangoutsMeet"},
            }
        },
    }

    try:
        created = svc.events().insert(
            calendarId=cal_id,
            body=event,
            conferenceDataVersion=1,
            sendUpdates="all",  # sends the invite email
        ).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Google create event error: {e}")

    return {
        "ok": True,
        "eventLink": created.get("htmlLink"),
        "hangoutLink": created.get("hangoutLink"),
        "start": event["start"],
        "end": event["end"],
        "attendees": attendees,
    }
