from __future__ import annotations

import json
import os
import zipfile
import io
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends, Header, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from motor.motor_asyncio import AsyncIOMotorClient
from jose import jwt
from passlib.context import CryptContext

# MongoDB connection
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://kiroszemic_db_user:zemichael@movies.osvv5xi.mongodb.net/?appName=Movies"
)
DB_NAME = os.getenv("MONGO_DB_NAME", "touch_biometrics")
MANAGER_PASSWORD = os.getenv("MANAGER_PASSWORD", "admin123")  # Change this!
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-this")
JWT_ALGORITHM = "HS256"

# Initialize MongoDB with proper SSL/TLS settings
# MongoDB Atlas requires TLS by default for mongodb+srv connections
client = AsyncIOMotorClient(
    MONGO_URI,
    serverSelectionTimeoutMS=30000,
    connectTimeoutMS=30000,
    socketTimeoutMS=30000,
    tls=True,
    tlsAllowInvalidCertificates=False,
)

db = client[DB_NAME]
events_collection = db["events"]
sessions_collection = db["sessions"]

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

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


class ManagerLogin(BaseModel):
    password: str


app = FastAPI(title="Touch Biometrics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.on_event("startup")
async def startup_event():
    """Test MongoDB connection on startup"""
    try:
        await client.admin.command('ping')
        print("✅ MongoDB connection successful")
    except Exception as e:
        print(f"⚠️ MongoDB connection warning: {e}")
        print("⚠️ The app will still start, but database operations may fail.")
        print("⚠️ Please check:")
        print("   1. MongoDB Atlas IP whitelist (add 0.0.0.0/0 for Render)")
        print("   2. Connection string is correct")
        print("   3. Database user has proper permissions")


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("role") != "manager":
            raise HTTPException(status_code=403, detail="Not authorized")
        return payload
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.get("/manager", response_class=HTMLResponse)
async def manager_dashboard():
    """Serve the manager dashboard HTML"""
    dashboard_path = Path(__file__).parent / "manager_dashboard.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    raise HTTPException(status_code=404, detail="Dashboard not found")


@app.post("/manager/login")
async def manager_login(login: ManagerLogin):
    if login.password != MANAGER_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")
    token = jwt.encode({"role": "manager"}, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return {"token": token, "ok": True}


@app.post("/upload")
async def upload(payload: UploadPayload):
    # Store events as documents
    events_doc = {
        "session_id": payload.session.session_id,
        "user_id": payload.user.user_id,
        "user_name": payload.user.user_name,
        "uploaded_at": datetime.utcnow(),
        "events_csv": payload.events_csv,
        "events_json": payload.events_json,
        "total_events": len(payload.events_json),
    }
    await events_collection.insert_one(events_doc)

    # Store session summary
    session_doc = {
        "session_id": payload.session.session_id,
        "user_id": payload.user.user_id,
        "user_name": payload.user.user_name,
        "started_at_iso": payload.session.started_at_iso,
        "ended_at_iso": payload.session.ended_at_iso,
        "device_w": payload.session.device_w,
        "device_h": payload.session.device_h,
        "dpr": payload.session.dpr,
        "tasks_completed": payload.session.tasks_completed,
        "total_events": payload.session.total_events,
        "sessions_csv": payload.sessions_csv,
        "uploaded_at": datetime.utcnow(),
    }
    await sessions_collection.insert_one(session_doc)

    public_base = os.getenv("PUBLIC_BASE_URL")
    response = {
        "ok": True,
        "saved": {
            "session_id": payload.session.session_id,
            "user_id": payload.user.user_id,
        },
    }
    if public_base:
        base = public_base.rstrip("/")
        response["urls"] = {
            "events_csv": f"{base}/manager/files/{payload.user.user_id}/{payload.session.session_id}/events.csv",
            "events_json": f"{base}/manager/files/{payload.user.user_id}/{payload.session.session_id}/events.json",
            "sessions_csv": f"{base}/manager/files/{payload.user.user_id}/sessions.csv",
        }
    return response


@app.post("/delete_session")
async def delete_session(payload: DeletePayload):
    result_events = await events_collection.delete_many(
        {"session_id": payload.session_id, "user_id": payload.user_id}
    )
    result_sessions = await sessions_collection.delete_many(
        {"session_id": payload.session_id, "user_id": payload.user_id}
    )
    if result_events.deleted_count == 0 and result_sessions.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True, "deleted": result_events.deleted_count + result_sessions.deleted_count}


# Manager endpoints
@app.get("/manager/sessions")
async def list_sessions(manager: dict = Depends(verify_token)):
    cursor = sessions_collection.find({}).sort("uploaded_at", -1)
    sessions = await cursor.to_list(length=1000)
    for s in sessions:
        s["_id"] = str(s["_id"])
        if "uploaded_at" in s:
            s["uploaded_at"] = s["uploaded_at"].isoformat()
        if "started_at_iso" in s:
            s["started_at_iso"] = s["started_at_iso"].isoformat() if isinstance(s["started_at_iso"], datetime) else s["started_at_iso"]
        if "ended_at_iso" in s:
            s["ended_at_iso"] = s["ended_at_iso"].isoformat() if isinstance(s["ended_at_iso"], datetime) else s["ended_at_iso"]
    return {"sessions": sessions, "count": len(sessions)}


@app.get("/manager/files/{user_id}/{session_id}/events.csv")
async def download_events_csv(
    user_id: str, session_id: str, manager: dict = Depends(verify_token)
):
    doc = await events_collection.find_one(
        {"user_id": user_id, "session_id": session_id}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="File not found")
    return StreamingResponse(
        io.BytesIO(doc["events_csv"].encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="events_{session_id}.csv"'},
    )


@app.get("/manager/files/{user_id}/{session_id}/events.json")
async def download_events_json(
    user_id: str, session_id: str, manager: dict = Depends(verify_token)
):
    doc = await events_collection.find_one(
        {"user_id": user_id, "session_id": session_id}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="File not found")
    return JSONResponse(
        content=doc["events_json"],
        headers={"Content-Disposition": f'attachment; filename="events_{session_id}.json"'},
    )


@app.get("/manager/files/{user_id}/sessions.csv")
async def download_sessions_csv(user_id: str, manager: dict = Depends(verify_token)):
    doc = await sessions_collection.find_one({"user_id": user_id})
    if not doc:
        raise HTTPException(status_code=404, detail="File not found")
    return StreamingResponse(
        io.BytesIO(doc["sessions_csv"].encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="sessions_{user_id}.csv"'},
    )


@app.get("/manager/download-all")
async def download_all_csvs(manager: dict = Depends(verify_token)):
    """Download all CSV files as a zip archive"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Add all events CSVs
        async for doc in events_collection.find({}):
            user_id = doc["user_id"]
            session_id = doc["session_id"]
            zip_file.writestr(
                f"{user_id}/events_{session_id}.csv", doc["events_csv"]
            )
            zip_file.writestr(
                f"{user_id}/events_{session_id}.json",
                json.dumps(doc["events_json"], indent=2),
            )

        # Add all sessions CSVs
        async for doc in sessions_collection.find({}):
            user_id = doc["user_id"]
            zip_file.writestr(f"{user_id}/sessions.csv", doc["sessions_csv"])

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="all_touch_data.zip"'},
    )
