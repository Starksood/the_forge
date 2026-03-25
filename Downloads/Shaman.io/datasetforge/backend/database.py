"""SQLite database layer using SQLAlchemy ORM."""
import os
import shutil
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from sqlalchemy.orm import Session as SQLSession

from .models import (
    Session, Document, FrameworkAnalysis, Chunk, Triple, Relationship,
    SessionStatus, ChunkStatus, TripleStatus,
    create_database_engine, create_session_factory, init_database
)

SESSIONS_DIR = "sessions"


class DatabaseManager:
    """Manages SQLAlchemy database connections and operations."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.database_url = f"sqlite:///{db_path}"
        self.engine = create_database_engine(self.database_url)
        self.session_factory = create_session_factory(self.engine)
        
        # Initialize database tables
        init_database(self.database_url)
    
    @contextmanager
    def get_session(self):
        """Context manager for database sessions."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def create_session_record(self, session_id: str, document_name: str, 
                            settings: Optional[Dict[str, Any]] = None) -> str:
        """Create a new session record and return the session ID."""
        with self.get_session() as db_session:
            session = Session(
                id=session_id,
                document_name=document_name,
                status=SessionStatus.UPLOADING.value,
                settings_json=settings or {}
            )
            db_session.add(session)
            db_session.commit()
            return session_id
    
    def get_session_record(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID as a dictionary."""
        with self.get_session() as db_session:
            session = db_session.query(Session).filter(Session.id == session_id).first()
            if session:
                return {
                    'id': session.id,
                    'document_name': session.document_name,
                    'created_at': session.created_at,
                    'updated_at': session.updated_at,
                    'status': session.status,
                    'framework_analysis_json': session.framework_analysis_json,
                    'settings_json': session.settings_json
                }
            return None
    
    def update_session_status(self, session_id: str, status: SessionStatus):
        """Update session status."""
        with self.get_session() as db_session:
            session = db_session.query(Session).filter(Session.id == session_id).first()
            if session:
                session.status = status.value
                session.updated_at = datetime.utcnow()
                db_session.commit()
    
    def create_document(self, document_id: str, session_id: str, filename: str, 
                       content: str, page_count: Optional[int] = None) -> str:
        """Create a new document record and return the document ID."""
        with self.get_session() as db_session:
            document = Document(
                id=document_id,
                session_id=session_id,
                filename=filename,
                content=content,
                page_count=page_count
            )
            db_session.add(document)
            db_session.commit()
            return document_id
    
    def create_framework_analysis(self, analysis_id: str, document_id: str,
                                 taxonomy: List[Dict], framework_summary: str,
                                 relationships: List[Dict], raw_analysis: str,
                                 compressed_context: Optional[str] = None) -> str:
        """Create a framework analysis record and return the analysis ID."""
        with self.get_session() as db_session:
            analysis = FrameworkAnalysis(
                id=analysis_id,
                document_id=document_id,
                taxonomy_json=taxonomy,
                framework_summary=framework_summary,
                compressed_context=compressed_context,
                relationships_json=relationships,
                raw_analysis=raw_analysis
            )
            db_session.add(analysis)
            db_session.commit()
            return analysis_id
    
    def create_chunk(self, chunk_id: str, document_id: str, sequence_number: int,
                    start_char: int, end_char: int, content: str, 
                    page_estimate: Optional[float] = None) -> str:
        """Create a chunk record and return the chunk ID."""
        with self.get_session() as db_session:
            chunk = Chunk(
                id=chunk_id,
                document_id=document_id,
                sequence_number=sequence_number,
                start_char=start_char,
                end_char=end_char,
                content=content,
                page_estimate=page_estimate,
                status=ChunkStatus.PENDING.value
            )
            db_session.add(chunk)
            db_session.commit()
            return chunk_id
    
    def create_triple(self, triple_id: str, chunk_id: str, angle: str, intensity: str,
                     system_prompt: str = "", user_message: str = "", 
                     assistant_response: str = "", thinking_trace: str = "",
                     tags: Optional[List[str]] = None, is_cross_reference: bool = False) -> str:
        """Create a triple record and return the triple ID."""
        with self.get_session() as db_session:
            triple = Triple(
                id=triple_id,
                chunk_id=chunk_id,
                angle=angle,
                intensity=intensity,
                system_prompt=system_prompt,
                user_message=user_message,
                assistant_response=assistant_response,
                thinking_trace=thinking_trace,
                tags_json=tags or [],
                is_cross_reference=is_cross_reference,
                status=TripleStatus.PENDING.value
            )
            db_session.add(triple)
            db_session.commit()
            return triple_id
    
    def update_triple(self, triple_id: str, **kwargs) -> bool:
        """Update a triple record and return success status."""
        with self.get_session() as db_session:
            triple = db_session.query(Triple).filter(Triple.id == triple_id).first()
            if triple:
                for key, value in kwargs.items():
                    if hasattr(triple, key):
                        setattr(triple, key, value)
                triple.updated_at = datetime.utcnow()
                db_session.commit()
                return True
            return False
    
    def get_chunks_by_document(self, document_id: str) -> List[Chunk]:
        """Get all chunks for a document."""
        with self.get_session() as db_session:
            return db_session.query(Chunk).filter(
                Chunk.document_id == document_id
            ).order_by(Chunk.sequence_number).all()
    
    def get_triples_by_chunk(self, chunk_id: str) -> List[Triple]:
        """Get all triples for a chunk."""
        with self.get_session() as db_session:
            return db_session.query(Triple).filter(Triple.chunk_id == chunk_id).all()
    
    def get_approved_triples(self, session_id: str) -> List[Triple]:
        """Get all approved triples for a session."""
        with self.get_session() as db_session:
            return db_session.query(Triple).join(Chunk).join(Document).filter(
                Document.session_id == session_id,
                Triple.status == TripleStatus.APPROVED.value
            ).all()
    
    def count_triples_by_status(self, session_id: str) -> Dict[str, int]:
        """Count triples by status for a session."""
        with self.get_session() as db_session:
            from sqlalchemy import func
            results = db_session.query(
                Triple.status, 
                func.count(Triple.id)
            ).join(Chunk).join(Document).filter(
                Document.session_id == session_id
            ).group_by(Triple.status).all()
            return {status: count for status, count in results}


def list_sessions() -> List[Dict[str, Any]]:
    """List all available session files."""
    sessions = []
    if not os.path.exists(SESSIONS_DIR):
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        return sessions
    
    for fname in os.listdir(SESSIONS_DIR):
        if fname.endswith(".db") and "_backup" not in fname:
            path = os.path.join(SESSIONS_DIR, fname)
            mtime = os.path.getmtime(path)
            sessions.append({
                "filename": fname,
                "path": path,
                "last_opened": datetime.fromtimestamp(mtime).isoformat(),
                "size": os.path.getsize(path),
            })
    
    sessions.sort(key=lambda x: x["last_opened"], reverse=True)
    return sessions


def get_session_summary(db_path: str) -> Dict[str, Any]:
    """Get summary information for a session."""
    try:
        db_manager = DatabaseManager(db_path)
        
        # Extract session ID from path
        session_filename = os.path.basename(db_path)
        session_id = session_filename.replace('.db', '')
        
        with db_manager.get_session() as db_session:
            # Get session record
            session_record = db_session.query(Session).filter(Session.id == session_id).first()
            if not session_record:
                return {"approved": 0, "document_name": "unknown"}
            
            # Count approved triples
            approved_count = db_session.query(Triple).join(Chunk).join(Document).filter(
                Document.session_id == session_id,
                Triple.status == TripleStatus.APPROVED.value
            ).count()
            
            return {
                "approved": approved_count,
                "document_name": session_record.document_name
            }
    except Exception:
        return {"approved": 0, "document_name": "unknown"}


def maybe_backup(db_path: str):
    """Create backup every 50 approvals."""
    try:
        db_manager = DatabaseManager(db_path)
        
        # Extract session ID from path
        session_filename = os.path.basename(db_path)
        session_id = session_filename.replace('.db', '')
        
        with db_manager.get_session() as db_session:
            # Count approved triples
            approved_count = db_session.query(Triple).join(Chunk).join(Document).filter(
                Document.session_id == session_id,
                Triple.status == TripleStatus.APPROVED.value
            ).count()
            
            # Backup every 50 approvals
            if approved_count > 0 and approved_count % 50 == 0:
                ts = int(time.time())
                base = db_path.replace(".db", "")
                backup_path = f"{base}_{ts}_backup.db"
                shutil.copy2(db_path, backup_path)
                
    except Exception:
        # Silently fail backup attempts
        pass


def create_session_database(session_id: str, document_name: str, 
                          settings: Optional[Dict[str, Any]] = None) -> str:
    """Create a new session database file."""
    if not os.path.exists(SESSIONS_DIR):
        os.makedirs(SESSIONS_DIR, exist_ok=True)
    
    db_path = os.path.join(SESSIONS_DIR, f"{session_id}.db")
    
    # Create database and session record
    db_manager = DatabaseManager(db_path)
    db_manager.create_session_record(session_id, document_name, settings)
    
    return db_path


def get_database_manager(db_path: str) -> DatabaseManager:
    """Get a DatabaseManager instance for the given path."""
    return DatabaseManager(db_path)