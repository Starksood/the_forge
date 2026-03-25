"""SQLAlchemy ORM models for DatasetForge database schema."""
import json
from datetime import datetime
from enum import Enum
from typing import List, Optional

from sqlalchemy import (
    Column, String, Integer, Text, Float, DateTime, ForeignKey, 
    Boolean, JSON, UniqueConstraint, create_engine, event
)
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from sqlalchemy.types import TypeDecorator, VARCHAR

Base = declarative_base()


class Angle(Enum):
    """Five emotional perspectives for triple generation."""
    FIRST_ENCOUNTER = "first_encounter"
    IDENTIFICATION = "identification"
    MAXIMUM_RESISTANCE = "maximum_resistance"
    YIELDING = "yielding"
    INTEGRATION = "integration"


class Intensity(Enum):
    """Two intensity levels for triple generation."""
    ACUTE = "acute"      # fragmented, overwhelming
    MODERATE = "moderate"  # coherent but afraid


class TripleStatus(Enum):
    """Status of generated triples."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_MANUAL = "needs_manual"


class SessionStatus(Enum):
    """Processing status of sessions."""
    UPLOADING = "uploading"
    ANALYZING = "analyzing"
    CHUNKING = "chunking"
    GENERATING = "generating"
    REVIEWING = "reviewing"
    COMPLETE = "complete"


class ChunkStatus(Enum):
    """Status of document chunks."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    MANUAL_SPLIT = "manual_split"


class JSONType(TypeDecorator):
    """JSON type for SQLite compatibility."""
    impl = VARCHAR
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return value


class Session(Base):
    """Session represents a complete document processing workflow."""
    __tablename__ = 'sessions'

    id = Column(String, primary_key=True)
    document_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = Column(String, nullable=False, default=SessionStatus.UPLOADING.value)
    framework_analysis_json = Column(JSONType)
    settings_json = Column(JSONType)

    # Relationships
    documents = relationship("Document", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Session(id='{self.id}', document_name='{self.document_name}', status='{self.status}')>"


class Document(Base):
    """Document represents an uploaded source document."""
    __tablename__ = 'documents'

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey('sessions.id'), nullable=False)
    filename = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    page_count = Column(Integer)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    session = relationship("Session", back_populates="documents")
    framework_analyses = relationship("FrameworkAnalysis", back_populates="document", cascade="all, delete-orphan")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Document(id='{self.id}', filename='{self.filename}', page_count={self.page_count})>"


class FrameworkAnalysis(Base):
    """Framework analysis contains extracted taxonomy and relationships."""
    __tablename__ = 'framework_analyses'

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey('documents.id'), nullable=False)
    taxonomy_json = Column(JSONType)
    framework_summary = Column(Text)
    compressed_context = Column(Text)
    relationships_json = Column(JSONType)
    raw_analysis = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="framework_analyses")
    relationships = relationship("Relationship", back_populates="framework_analysis", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<FrameworkAnalysis(id='{self.id}', document_id='{self.document_id}')>"


class Chunk(Base):
    """Chunk represents a semantically coherent section of a document."""
    __tablename__ = 'chunks'

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey('documents.id'), nullable=False)
    sequence_number = Column(Integer, nullable=False)
    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    page_estimate = Column(Float)
    status = Column(String, nullable=False, default=ChunkStatus.PENDING.value)

    # Relationships
    document = relationship("Document", back_populates="chunks")
    triples = relationship("Triple", back_populates="chunk", cascade="all, delete-orphan")

    # Constraints
    __table_args__ = (
        UniqueConstraint('document_id', 'sequence_number', name='_document_sequence_uc'),
    )

    def __repr__(self):
        return f"<Chunk(id='{self.id}', sequence_number={self.sequence_number}, status='{self.status}')>"


class Triple(Base):
    """Triple represents a {system, user, assistant} conversation unit."""
    __tablename__ = 'triples'

    id = Column(String, primary_key=True)
    chunk_id = Column(String, ForeignKey('chunks.id'), nullable=False)
    angle = Column(String, nullable=False)
    intensity = Column(String, nullable=False)
    system_prompt = Column(Text)
    user_message = Column(Text)
    assistant_response = Column(Text)
    thinking_trace = Column(Text)
    status = Column(String, nullable=False, default=TripleStatus.PENDING.value)
    tags_json = Column(JSONType)
    is_cross_reference = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    chunk = relationship("Chunk", back_populates="triples")

    @property
    def tags(self) -> List[str]:
        """Get tags as a list."""
        return self.tags_json or []

    @tags.setter
    def tags(self, value: List[str]):
        """Set tags from a list."""
        self.tags_json = value

    def __repr__(self):
        return f"<Triple(id='{self.id}', angle='{self.angle}', intensity='{self.intensity}', status='{self.status}')>"


class Relationship(Base):
    """Relationship represents conceptual connections between document sections."""
    __tablename__ = 'relationships'

    id = Column(String, primary_key=True)
    framework_id = Column(String, ForeignKey('framework_analyses.id'), nullable=False)
    source_concept = Column(String, nullable=False)
    target_concept = Column(String, nullable=False)
    relationship_type = Column(String, nullable=False)
    description = Column(Text)

    # Relationships
    framework_analysis = relationship("FrameworkAnalysis", back_populates="relationships")

    def __repr__(self):
        return f"<Relationship(id='{self.id}', source='{self.source_concept}', target='{self.target_concept}')>"


def create_database_engine(database_url: str):
    """Create SQLAlchemy engine with SQLite optimizations."""
    engine = create_engine(
        database_url,
        echo=False,  # Set to True for SQL debugging
        connect_args={
            "check_same_thread": False,  # Allow multiple threads
            "timeout": 30,  # Connection timeout
        },
        pool_pre_ping=True,  # Verify connections before use
        pool_recycle=3600,  # Recycle connections every hour
    )
    
    # Configure SQLite for optimal performance
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        # Enable WAL mode for concurrent access
        cursor.execute("PRAGMA journal_mode=WAL")
        # Optimize for performance
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=1000")
        cursor.execute("PRAGMA temp_store=memory")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    
    return engine


def create_session_factory(engine):
    """Create SQLAlchemy session factory."""
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_database(database_url: str):
    """Initialize database with all tables."""
    engine = create_database_engine(database_url)
    Base.metadata.create_all(engine)
    return engine