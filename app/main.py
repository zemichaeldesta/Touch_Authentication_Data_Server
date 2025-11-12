from __future__ import annotations
import os
import json
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

DATA_ROOT = Path("data")
DATA_ROOT.mkdir(exist_ok=True, parents=True)

ALLOWED_ORIGINS_RAW = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS: List[str] = [
    origin.strip() for origin in ALLOWED_ORIGINS_RAW.split(",") if origin.strip()
]


EVENTS_HEADER = (
    "session_id,user_id,task,phase,t_ms,type,x,y,pressure,tiltX,tiltY,extra,json"
)
SESSIONS_HEADER = (
    "session_id,user_id,started_at_iso,ended_at_iso,device_w,device_h,dpr,"
    "tasks_completed,total_events"
)


class ScreenSpec(BaseModel):
    w: int
    h: int
    dpr: float


class UserPayload(BaseModel):
    user_id: str = Field(..., min_length=6)
    user_name: str = Field(..., min_length=3)
    tz: str
    ua: str
    screen: ScreenSpec


class SessionPayload(BaseModel):
    session_id: str = Field(..., min_length=6)
    started_at_iso: datetime
    ended_at_iso: datetime | None = None
    device_w: int
    device_h: int
    dpr: float
    tasks_completed: int
    total_events: int


class UploadPayload(BaseModel):
    user: UserPayload
    session: SessionPayload
    events_csv: str = Field(..., min_length=1)
    sessions_csv: str = Field(..., min_length=1)
    events_json: List[dict]

    @validator("events_json")
    def validate_event_volume(cls, value: List[dict]):
        if len(value) > 200_000:
            raise ValueError("Too many events in payload")
        return value

    @validator("events_csv")
    def validate_events_header(cls, value: str):
        header = value.strip().splitlines()[0]
        if header != EVENTS_HEADER:
            raise ValueError("Invalid events.csv header")
        return value

    @validator("sessions_csv")
    def validate_sessions_header(cls, value: str):
        header = value.strip().splitlines()[0]
        if header != SESSIONS_HEADER:
            raise ValueError("Invalid sessions.csv header")
        return value


class DeletePayload(BaseModel):
    session_id: str
    user_id: str


app = FastAPI(title="Touch Biometrics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


def ensure_directory(user_id: str) -> Path:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    target_dir = DATA_ROOT / today / user_id
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


@app.post("/upload")
async def upload(payload: UploadPayload):
    directory = ensure_directory(payload.user.user_id)
    events_path = directory / f"events_{payload.session.session_id}.csv"
    events_json_path = directory / f"events_{payload.session.session_id}.json"
    sessions_path = directory / "sessions.csv"

    events_path.write_text(payload.events_csv, encoding="utf-8")
    events_json_path.write_text(
        json.dumps(payload.events_json, ensure_ascii=False), encoding="utf-8"
    )

    session_lines = payload.sessions_csv.strip().splitlines()
    if not session_lines:
        raise HTTPException(status_code=400, detail="sessions_csv is empty")

    if sessions_path.exists():
        with sessions_path.open("a", encoding="utf-8") as handle:
            for line in session_lines[1:]:
                handle.write(f"{line}\n")
    else:
        sessions_path.write_text(
            payload.sessions_csv.strip() + "\n", encoding="utf-8"
        )

    return {
        "ok": True,
        "saved": {
            "events_csv": str(events_path),
            "sessions_csv": str(sessions_path),
            "events_json": str(events_json_path),
        },
    }


@app.post("/delete_session")
async def delete_session(payload: DeletePayload):
    matches = list(DATA_ROOT.glob(f"*/{payload.user_id}"))
    if not matches:
        raise HTTPException(status_code=404, detail="Session not found")
    session_dir = matches[0]
    removed = []
    for file in session_dir.glob(f"*{payload.session_id}*"):
        file.unlink(missing_ok=True)
        removed.append(file.name)
    if not removed:
        raise HTTPException(status_code=404, detail="Session files not found")
    return {"ok": True, "removed": removed}

