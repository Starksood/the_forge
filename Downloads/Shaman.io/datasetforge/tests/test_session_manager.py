"""Unit tests for SessionManager class."""
import os
import tempfile
import shutil
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from backend.session_manager import SessionManager, SessionInfo
from backend.database import SESSIONS_DIR
from backend.models import SessionStatus


class TestSessionManager:
    """Test cases for SessionManager class."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Create temporary directory for test sessions
        self.temp_dir = tempfile.mkdtemp()
        self.original_sessions_dir = SESSIONS_DIR
        
        # Patch SESSIONS_DIR to use temp directory
        self.sessions_dir_patcher = patch('backend.session_manager.SESSIONS_DIR', self.temp_dir)
        self.sessions_dir_patcher.start()
        
        # Also patch in database module
        self.db_sessions_dir_patcher = patch('backend.database.SESSIONS_DIR', self.temp_dir)
        self.db_sessions_dir_patcher.start()
        
        self.session_manager = SessionManager()
    
    def teardown_method(self):
        """Clean up after each test."""
        self.sessions_dir_patcher.stop()
        self.db_sessions_dir_patcher.stop()
        
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_create_session_generates_unique_id(self):
        """Test that create_session generates unique timestamp-based IDs."""
        document_name = "test_document.pdf"
        
        # Create two sessions
        session_id1 = self.session_manager.create_session(document_name)
        session_id2 = self.session_manager.create_session(document_name)
        
        # IDs should be different
        assert session_id1 != session_id2
        
        # Both should contain timestamp format
        assert len(session_id1.split('_')) >= 3  # YYYYMMDD_HHMMSS_hash
        assert len(session_id2.split('_')) >= 3
        
        # Database files should exist
        db_path1 = os.path.join(self.temp_dir, f"{session_id1}.db")
        db_path2 = os.path.join(self.temp_dir, f"{session_id2}.db")
        
        assert os.path.exists(db_path1)
        assert os.path.exists(db_path2)
    
    def test_create_session_with_settings(self):
        """Test creating session with custom settings."""
        document_name = "test_document.pdf"
        settings = {
            "ollama_host": "http://localhost:11434",
            "model_name": "llama3.2:3b",
            "triples_per_chunk": 10
        }
        
        session_id = self.session_manager.create_session(document_name, settings)
        
        # Load session and verify settings
        db_manager = self.session_manager.load_session(session_id)
        session_record = db_manager.get_session_record(session_id)
        
        assert session_record is not None
        assert session_record["document_name"] == document_name
        assert session_record["settings_json"] == settings
    
    def test_load_session_success(self):
        """Test loading an existing session."""
        document_name = "test_document.pdf"
        session_id = self.session_manager.create_session(document_name)
        
        # Load the session
        db_manager = self.session_manager.load_session(session_id)
        
        assert db_manager is not None
        
        # Verify session data
        session_record = db_manager.get_session_record(session_id)
        assert session_record["document_name"] == document_name
        assert session_record["status"] == SessionStatus.UPLOADING.value
    
    def test_load_session_not_found(self):
        """Test loading a non-existent session raises FileNotFoundError."""
        non_existent_id = "20240101_120000_nonexistent"
        
        with pytest.raises(FileNotFoundError):
            self.session_manager.load_session(non_existent_id)
    
    def test_session_exists(self):
        """Test session existence checking."""
        document_name = "test_document.pdf"
        session_id = self.session_manager.create_session(document_name)
        
        # Session should exist
        assert self.session_manager.session_exists(session_id) is True
        
        # Non-existent session should not exist
        assert self.session_manager.session_exists("nonexistent") is False
    
    def test_get_session_path(self):
        """Test getting session database path."""
        session_id = "20240101_120000_test"
        expected_path = os.path.join(self.temp_dir, f"{session_id}.db")
        
        actual_path = self.session_manager.get_session_path(session_id)
        
        assert actual_path == expected_path
    
    def test_delete_session_success(self):
        """Test successful session deletion."""
        document_name = "test_document.pdf"
        session_id = self.session_manager.create_session(document_name)
        
        # Verify session exists
        assert self.session_manager.session_exists(session_id) is True
        
        # Delete session
        result = self.session_manager.delete_session(session_id)
        
        assert result is True
        assert self.session_manager.session_exists(session_id) is False
    
    def test_delete_session_not_found(self):
        """Test deleting non-existent session returns False."""
        result = self.session_manager.delete_session("nonexistent")
        assert result is False
    
    def test_get_available_sessions_empty(self):
        """Test getting available sessions when none exist."""
        sessions = self.session_manager.get_available_sessions()
        assert sessions == []
    
    def test_get_available_sessions_with_data(self):
        """Test getting available sessions with existing data."""
        # Create test sessions
        doc1 = "document1.pdf"
        doc2 = "document2.pdf"
        
        session_id1 = self.session_manager.create_session(doc1)
        session_id2 = self.session_manager.create_session(doc2)
        
        # Get available sessions
        sessions = self.session_manager.get_available_sessions()
        
        assert len(sessions) == 2
        
        # Verify session info
        session_ids = [s.id for s in sessions]
        assert session_id1 in session_ids
        assert session_id2 in session_ids
        
        # Find specific session
        session1_info = next(s for s in sessions if s.id == session_id1)
        assert session1_info.document_name == doc1
        assert session1_info.status == SessionStatus.UPLOADING.value
        assert session1_info.approved_count == 0
    
    @patch('backend.session_manager.maybe_backup')
    def test_auto_backup_calls_maybe_backup(self, mock_maybe_backup):
        """Test that auto_backup calls maybe_backup with correct path."""
        document_name = "test_document.pdf"
        session_id = self.session_manager.create_session(document_name)
        
        # Call auto_backup
        self.session_manager.auto_backup(session_id)
        
        # Verify maybe_backup was called with correct path
        expected_path = os.path.join(self.temp_dir, f"{session_id}.db")
        mock_maybe_backup.assert_called_once_with(expected_path)
    
    def test_auto_backup_nonexistent_session(self):
        """Test auto_backup with non-existent session doesn't crash."""
        # Should not raise exception
        self.session_manager.auto_backup("nonexistent")
    
    def test_get_session_stats_success(self):
        """Test getting session statistics."""
        document_name = "test_document.pdf"
        session_id = self.session_manager.create_session(document_name)
        
        # Get stats
        stats = self.session_manager.get_session_stats(session_id)
        
        assert stats["session_id"] == session_id
        assert stats["document_name"] == document_name
        assert stats["status"] == SessionStatus.UPLOADING.value
        assert stats["total_triples"] == 0
        assert stats["approved_triples"] == 0
        assert isinstance(stats["created_at"], datetime)
        assert isinstance(stats["updated_at"], datetime)
    
    def test_get_session_stats_nonexistent(self):
        """Test getting stats for non-existent session returns empty dict."""
        stats = self.session_manager.get_session_stats("nonexistent")
        assert stats == {}
    
    def test_sessions_directory_creation(self):
        """Test that SessionManager creates sessions directory if it doesn't exist."""
        # Remove temp directory
        shutil.rmtree(self.temp_dir)
        
        # Create new SessionManager - should recreate directory
        session_manager = SessionManager()
        
        assert os.path.exists(self.temp_dir)
    
    def test_timestamp_based_naming_format(self):
        """Test that session IDs follow timestamp-based naming convention."""
        document_name = "test_document.pdf"
        session_id = self.session_manager.create_session(document_name)
        
        # Should have format: YYYYMMDD_HHMMSS_hash
        parts = session_id.split('_')
        assert len(parts) == 3
        
        # First part should be date (YYYYMMDD)
        date_part = parts[0]
        assert len(date_part) == 8
        assert date_part.isdigit()
        
        # Second part should be time (HHMMSS)
        time_part = parts[1]
        assert len(time_part) == 6
        assert time_part.isdigit()
        
        # Third part should be hash (8 characters)
        hash_part = parts[2]
        assert len(hash_part) == 8
    
    def test_never_overwrite_existing_sessions(self):
        """Test that SessionManager never overwrites existing session databases."""
        document_name = "test_document.pdf"
        
        # Create first session
        session_id1 = self.session_manager.create_session(document_name)
        db_path1 = self.session_manager.get_session_path(session_id1)
        
        # Verify first session exists and load its data
        assert os.path.exists(db_path1)
        db_manager1 = self.session_manager.load_session(session_id1)
        session_record1 = db_manager1.get_session_record(session_id1)
        original_created_at = session_record1["created_at"]
        
        # Create second session (should have different ID)
        session_id2 = self.session_manager.create_session(document_name)
        
        # First session should be unchanged (same creation time, still exists)
        assert os.path.exists(db_path1)
        db_manager1_after = self.session_manager.load_session(session_id1)
        session_record1_after = db_manager1_after.get_session_record(session_id1)
        assert session_record1_after["created_at"] == original_created_at
        assert session_id1 != session_id2
        
        # Both sessions should exist
        assert self.session_manager.session_exists(session_id1)
        assert self.session_manager.session_exists(session_id2)


if __name__ == "__main__":
    pytest.main([__file__])