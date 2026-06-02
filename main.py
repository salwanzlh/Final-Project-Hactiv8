"""
Showroom AI — Backend
FastAPI + WebSocket

Jalankan:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000

Docs tersedia di:
  http://localhost:8000/docs
"""
import uuid
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from backend.config import settings
from backend.routers import ws as ws_router
from backend.routers import cars as cars_router

# ── Logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚗 Showroom AI starting — mode: {settings.app_mode.upper()}")
    yield
    logger.info("Showroom AI shutting down.")


# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title="Showroom AI — Backend",
    description="Real-time AI assistant untuk sales mobil via WebSocket.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────
app.include_router(ws_router.router)
app.include_router(cars_router.router)


# ── Health & utility endpoints ────────────────────────────
@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "mode": settings.app_mode}


@app.get("/api/new-session", tags=["system"])
def new_session():
    """Frontend panggil endpoint ini untuk mendapatkan session_id baru."""
    return {"session_id": str(uuid.uuid4())}


@app.get("/metrics", tags=["system"], include_in_schema=False)
def metrics():
    """Endpoint Prometheus — di-scrape oleh prometheus server."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
