"""
SessionManager — mengelola semua koneksi WebSocket aktif.
Setiap sales punya satu session_id unik per sesi percakapan.
"""
import uuid
import json
import logging
from datetime import datetime, timezone
from fastapi import WebSocket
from backend.models.schemas import ConversationContext, WsOutgoing

logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(self):
        # session_id → WebSocket
        self.active: dict[str, WebSocket] = {}
        # session_id → ConversationContext
        self.contexts: dict[str, ConversationContext] = {}

    def new_session_id(self) -> str:
        return str(uuid.uuid4()) # Generate unique session ID

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept() # Terima koneksi WebSocket dari sales
        self.active[session_id] = websocket # Simpan koneksi WebSocket sesuai session_id
        self.contexts[session_id] = ConversationContext(session_id=session_id) # simpan context 
        logger.info(f"[WS] Session connected: {session_id}")

    def disconnect(self, session_id: str):
        self.active.pop(session_id, None)
        # Simpan context untuk evaluasi — jangan langsung dihapus
        logger.info(f"[WS] Session disconnected: {session_id}")

    async def send(self, session_id: str, msg_type: str, payload: dict):
        ws = self.active.get(session_id)
        if not ws:
            return
        out = WsOutgoing(
            type=msg_type,
            payload=payload,
            session_id=session_id,
            ts=datetime.now(timezone.utc),
        )
        try:
            await ws.send_text(out.model_dump_json())
        except Exception as e:
            logger.warning(f"[WS] Send failed for {session_id}: {e}")
            self.disconnect(session_id)

    async def broadcast(self, msg_type: str, payload: dict):
        """Kirim pesan ke semua sesi aktif (misal: notifikasi promo)."""
        for session_id in list(self.active.keys()):
            await self.send(session_id, msg_type, payload)

    def get_context(self, session_id: str) -> ConversationContext | None:
        return self.contexts.get(session_id)


# Singleton — diimpor oleh router dan services
session_manager = SessionManager()
