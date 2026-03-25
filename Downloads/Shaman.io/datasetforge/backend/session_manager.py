"""SessionManager provides high-level interface for managing DatasetForge sessions."""
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .database import (
    DatabaseManager, create_session_database, list_sessions, 
    get_session_summary, maybe_backup, SESSIONS_DIR
)
from .models import SessionStatus


@dataclass
class SessionInfo:
    """Information about an available session."""
    id: str
    document_name: str
    created_at: datetime
    updated_at: datetime
    status: str
    filename: str
    path: str
    size: int
    approved_count: int


class SessionManager:
    """High-level interface for managing DatasetForge sessions."""
    
    def __init__(self):
        """Initialize SessionManager."""
        # Ensure sessions directory exists
        if not os.path.exists(SESSIONS_DIR):
            os.makedirs(SESSIONS_DIR, exist_ok=True)
    
    def create_session(self, document_name: str, settings: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new session with timestamp-based naming.
        
        Args:
            document_name: Name of the document being processed
            settings: Optional session settings
            
        Returns:
            Session ID of the created session
        """
        # Generate timestamp-based session ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        
        # Create session database
        db_path = create_session_database(session_id, document_name, settings)
        
        return session_id
    
    def load_session(self, session_id: str) -> DatabaseManager:
        """
        Load an existing session by ID.
        
        Args:
            session_id: ID of the session to load
            
        Returns:
            DatabaseManager instance for the session
            
        Raises:
            FileNotFoundError: If session database doesn't exist
        """
        db_path = os.path.join(SESSIONS_DIR, f"{session_id}.db")
        
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Session database not found: {db_path}")
        
        return DatabaseManager(db_path)
    
    def get_available_sessions(self) -> List[SessionInfo]:
        """
        Get list of all available sessions with metadata.
        
        Returns:
            List of SessionInfo objects sorted by last modified (newest first)
        """
        session_files = list_sessions()
        session_infos = []
        
        for session_file in session_files:
            try:
                # Extract session ID from filename
                session_id = os.path.basename(session_file["filename"]).replace('.db', '')
                
                # Get session summary
                summary = get_session_summary(session_file["path"])
                
                # Load session record for detailed info
                db_manager = DatabaseManager(session_file["path"])
                session_record = db_manager.get_session_record(session_id)
                
                if session_record:
                    session_info = SessionInfo(
                        id=session_id,
                        document_name=session_record["document_name"],
                        created_at=session_record["created_at"],
                        updated_at=session_record["updated_at"],
                        status=session_record["status"],
                        filename=session_file["filename"],
                        path=session_file["path"],
                        size=session_file["size"],
                        approved_count=summary["approved"]
                    )
                    session_infos.append(session_info)
                    
            except Exception:
                # Skip corrupted or invalid session files
                continue
        
        # Sort by updated_at (newest first)
        session_infos.sort(key=lambda x: x.updated_at, reverse=True)
        
        return session_infos
    
    def auto_backup(self, session_id: str) -> None:
        """
        Create automatic backup if needed (every 50 approvals).
        
        Args:
            session_id: ID of the session to backup
        """
        db_path = os.path.join(SESSIONS_DIR, f"{session_id}.db")
        
        if os.path.exists(db_path):
            maybe_backup(db_path)
    
    def session_exists(self, session_id: str) -> bool:
        """
        Check if a session exists.
        
        Args:
            session_id: ID of the session to check
            
        Returns:
            True if session exists, False otherwise
        """
        db_path = os.path.join(SESSIONS_DIR, f"{session_id}.db")
        return os.path.exists(db_path)
    
    def get_session_path(self, session_id: str) -> str:
        """
        Get the database path for a session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            Full path to the session database file
        """
        return os.path.join(SESSIONS_DIR, f"{session_id}.db")
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and its database file.
        
        Args:
            session_id: ID of the session to delete
            
        Returns:
            True if session was deleted, False if it didn't exist
        """
        db_path = os.path.join(SESSIONS_DIR, f"{session_id}.db")
        
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
                return True
            except OSError:
                return False
        
        return False
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            Dictionary with session statistics
        """
        try:
            db_manager = self.load_session(session_id)
            session_record = db_manager.get_session_record(session_id)
            
            if not session_record:
                return {}
            
            # Get triple counts by status
            triple_counts = db_manager.count_triples_by_status(session_id)
            
            return {
                "session_id": session_id,
                "document_name": session_record["document_name"],
                "status": session_record["status"],
                "created_at": session_record["created_at"],
                "updated_at": session_record["updated_at"],
                "triple_counts": triple_counts,
                "total_triples": sum(triple_counts.values()),
                "approved_triples": triple_counts.get("approved", 0)
            }
            
        except Exception:
            return {}