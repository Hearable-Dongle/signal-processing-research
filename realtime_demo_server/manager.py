from __future__ import annotations

import threading

from fastapi import HTTPException

from .models import SessionStartRequest
from .live_session import LiveDemoSession
from .session import DemoSession

SessionHandle = DemoSession | LiveDemoSession


class SessionManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._session: SessionHandle | None = None

    def start_session(self, req: SessionStartRequest) -> SessionHandle:
        with self._lock:
            if self._session is not None and self._session.status in {"starting", "running"}:
                raise HTTPException(status_code=409, detail="A session is already running")
            if req.input_source == "respeaker_live":
                session = LiveDemoSession(req)
            else:
                session = DemoSession(req)
            self._session = session
            session.start()
            return session

    def get_session(self, session_id: str) -> SessionHandle:
        with self._lock:
            if self._session is None or self._session.session_id != session_id:
                raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}")
            return self._session

    def stop_session(self, session_id: str) -> SessionHandle:
        session = self.get_session(session_id)
        session.stop()
        return session


manager = SessionManager()
