"""Integration tests for SessionManager with database infrastructure."""
import os
import tempfile
import shutil
import pytest
from unittest.mock import patch

from backend.session_manager import SessionManager
from backend.database import DatabaseManager
from backend.models import SessionStatus, TripleStatus


class TestSessionIntegration:
    """Integration tests for SessionManager with database operations."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Create temporary directory for test sessions
        self.temp_dir = tempfile.mkdtemp()
        
        # Patch SESSIONS_DIR to use temp directory
        self.sessions_dir_patcher = patch('backend.session_manager.SESSIONS_DIR', self.temp_dir)
        self.sessions_dir_patcher.start()
        
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
    
    def test_session_manager_database_integration(self):
        """Test that SessionManager integrates correctly with DatabaseManager."""
        document_name = "integration_test.pdf"
        
        # Create session using SessionManager
        session_id = self.session_manager.create_session(document_name)
        
        # Load session using SessionManager
        db_manager = self.session_manager.load_session(session_id)
        
        # Verify session record exists and is correct
        session_record = db_manager.get_session_record(session_id)
        assert session_record is not None
        assert session_record["document_name"] == document_name
        assert session_record["status"] == SessionStatus.UPLOADING.value
        
        # Test updating session status through DatabaseManager
        db_manager.update_session_status(session_id, SessionStatus.ANALYZING)
        
        # Verify status update through SessionManager
        updated_stats = self.session_manager.get_session_stats(session_id)
        assert updated_stats["status"] == SessionStatus.ANALYZING.value
    
    def test_session_backup_integration(self):
        """Test that backup functionality works with real database operations."""
        document_name = "backup_test.pdf"
        
        # Create session
        session_id = self.session_manager.create_session(document_name)
        db_manager = self.session_manager.load_session(session_id)
        
        # Create some test data (document and chunks)
        document_id = "test_doc_1"
        db_manager.create_document(document_id, session_id, "test.pdf", "Test content", 1)
        
        chunk_id = "test_chunk_1"
        db_manager.create_chunk(chunk_id, document_id, 1, 0, 100, "Test chunk content")
        
        # Create exactly 50 approved triples to trigger backup
        for i in range(50):
            triple_id = f"test_triple_{i}"
            db_manager.create_triple(
                triple_id, chunk_id, "first_encounter", "acute",
                "System prompt", "User message", "Assistant response"
            )
            # Approve the triple
            db_manager.update_triple(triple_id, status=TripleStatus.APPROVED.value)
        
        # Trigger backup
        self.session_manager.auto_backup(session_id)
        
        # Check if backup file was created
        backup_files = [f for f in os.listdir(self.temp_dir) if "_backup.db" in f]
        assert len(backup_files) > 0, "Backup file should have been created"
    
    def test_session_stats_with_real_data(self):
        """Test session statistics with real database data."""
        document_name = "stats_test.pdf"
        
        # Create session and add data
        session_id = self.session_manager.create_session(document_name)
        db_manager = self.session_manager.load_session(session_id)
        
        # Create document and chunk
        document_id = "stats_doc_1"
        db_manager.create_document(document_id, session_id, "stats.pdf", "Stats content", 1)
        
        chunk_id = "stats_chunk_1"
        db_manager.create_chunk(chunk_id, document_id, 1, 0, 100, "Stats chunk content")
        
        # Create triples with different statuses
        approved_count = 3
        rejected_count = 2
        pending_count = 1
        
        for i in range(approved_count):
            triple_id = f"approved_triple_{i}"
            db_manager.create_triple(triple_id, chunk_id, "first_encounter", "acute")
            db_manager.update_triple(triple_id, status=TripleStatus.APPROVED.value)
        
        for i in range(rejected_count):
            triple_id = f"rejected_triple_{i}"
            db_manager.create_triple(triple_id, chunk_id, "identification", "moderate")
            db_manager.update_triple(triple_id, status=TripleStatus.REJECTED.value)
        
        for i in range(pending_count):
            triple_id = f"pending_triple_{i}"
            db_manager.create_triple(triple_id, chunk_id, "yielding", "acute")
            # Leave as pending
        
        # Get stats through SessionManager
        stats = self.session_manager.get_session_stats(session_id)
        
        assert stats["approved_triples"] == approved_count
        assert stats["total_triples"] == approved_count + rejected_count + pending_count
        assert stats["document_name"] == document_name
        assert stats["session_id"] == session_id
    
    def test_available_sessions_with_real_data(self):
        """Test getting available sessions with real database content."""
        # Create multiple sessions with different data
        sessions_data = [
            ("document1.pdf", 5),  # 5 approved triples
            ("document2.pdf", 0),  # 0 approved triples
            ("document3.pdf", 10), # 10 approved triples
        ]
        
        created_sessions = []
        
        for doc_name, approved_count in sessions_data:
            session_id = self.session_manager.create_session(doc_name)
            created_sessions.append(session_id)
            
            if approved_count > 0:
                # Add some approved triples
                db_manager = self.session_manager.load_session(session_id)
                
                document_id = f"doc_{session_id}"
                db_manager.create_document(document_id, session_id, doc_name, "Content", 1)
                
                chunk_id = f"chunk_{session_id}"
                db_manager.create_chunk(chunk_id, document_id, 1, 0, 100, "Chunk content")
                
                for i in range(approved_count):
                    triple_id = f"triple_{session_id}_{i}"
                    db_manager.create_triple(triple_id, chunk_id, "first_encounter", "acute")
                    db_manager.update_triple(triple_id, status=TripleStatus.APPROVED.value)
        
        # Get available sessions
        available_sessions = self.session_manager.get_available_sessions()
        
        assert len(available_sessions) == len(sessions_data)
        
        # Verify session data
        for session_info in available_sessions:
            assert session_info.id in created_sessions
            
            # Find expected data
            expected_doc_name = None
            expected_approved = None
            for doc_name, approved_count in sessions_data:
                if session_info.document_name == doc_name:
                    expected_doc_name = doc_name
                    expected_approved = approved_count
                    break
            
            assert expected_doc_name is not None
            assert session_info.approved_count == expected_approved


if __name__ == "__main__":
    pytest.main([__file__])