
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    
    def __init__(self, timeout_minutes: int = 30):
        self.sessions: Dict[str, Dict] = {}
        self.timeout = timeout_minutes
        logger.info(f"Session manager initialized with {timeout_minutes} minute timeout")
    
    def create_session(self, chatbot_instance, session_id: Optional[str] = None) -> str:
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        current_time = datetime.now()
        self.sessions[session_id] = {
            "created_at": current_time,
            "last_active": current_time,
            "chatbot": chatbot_instance
        }
        
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get a session by ID if it exists and is not expired."""
        if session_id not in self.sessions:
            logger.warning(f"Session not found: {session_id}")
            return None
        
        session = self.sessions[session_id]
        current_time = datetime.now()
        
        if current_time - session["last_active"] > timedelta(minutes=self.timeout):
            logger.info(f"Session expired: {session_id}")
            self.end_session(session_id)
            return None
        
        session["last_active"] = current_time
        return session
    
    def get_or_create_session(self, chatbot_factory, session_id: Optional[str] = None) -> Tuple[str, object]:
        """Get an existing session or create a new one with the chatbot factory function."""
        self._clean_expired_sessions()
        
        if session_id and session_id in self.sessions:
            session = self.get_session(session_id)
            if session:
                logger.debug(f"Returning existing session: {session_id}")
                return session_id, session["chatbot"]
        
        chatbot = chatbot_factory()
        
        new_session_id = self.create_session(chatbot, session_id)
        return new_session_id, chatbot
    
    def end_session(self, session_id: str) -> bool:
        """End a session by ID."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Session ended: {session_id}")
            return True
        return False
    
    def _clean_expired_sessions(self) -> int:
        """Remove expired sessions and return count of removed sessions."""
        current_time = datetime.now()
        expired_ids = [
            sid for sid, session in self.sessions.items()
            if current_time - session["last_active"] > timedelta(minutes=self.timeout)
        ]
        
        for sid in expired_ids:
            del self.sessions[sid]
        
        if expired_ids:
            logger.info(f"Cleaned {len(expired_ids)} expired sessions")
        
        return len(expired_ids)
    
    def get_active_session_count(self) -> int:
        """Get the count of active sessions."""
        self._clean_expired_sessions()
        return len(self.sessions)