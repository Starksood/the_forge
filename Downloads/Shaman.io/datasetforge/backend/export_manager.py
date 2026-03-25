"""ExportManager — JSONL export with duplicate detection and metadata."""
import json
import hashlib
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple

from .database import DatabaseManager
from .models import Triple, Chunk, Document, TripleStatus

logger = logging.getLogger(__name__)


class ExportManager:
    """
    Exports approved triples in JSONL format compatible with
    HuggingFace / Unsloth fine-tuning pipelines.

    Requirements: 9.1, 9.2, 9.3, 9.4, 9.6
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def get_export_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for an export preview."""
        counts = self.db_manager.count_triples_by_status(session_id)
        approved = counts.get("approved", 0)
        pending = counts.get("pending", 0)
        rejected = counts.get("rejected", 0)
        needs_manual = counts.get("needs_manual", 0)
        total = sum(counts.values())

        # Count duplicates among approved
        approved_triples = self._get_approved_triples(session_id)
        unique, duplicates = self._detect_duplicates(approved_triples)

        return {
            "total_triples": total,
            "approved": approved,
            "pending": pending,
            "rejected": rejected,
            "needs_manual": needs_manual,
            "exportable": len(unique),
            "duplicates_removed": len(duplicates),
        }

    def export_jsonl(self, session_id: str) -> Tuple[str, Dict[str, Any]]:
        """
        Export approved triples as JSONL string.

        Returns:
            Tuple of (jsonl_content, metadata_dict)
        """
        approved_triples = self._get_approved_triples(session_id)
        unique, duplicates = self._detect_duplicates(approved_triples)

        lines = []
        for triple_data in unique:
            record = {
                "conversations": [
                    {"role": "system", "content": triple_data["system_prompt"]},
                    {"role": "user", "content": triple_data["user_message"]},
                    {"role": "assistant", "content": triple_data["assistant_response"]},
                ]
            }
            lines.append(json.dumps(record, ensure_ascii=False))

        jsonl_content = "\n".join(lines) + "\n" if lines else ""

        # Angle / intensity breakdown
        angle_counts: Dict[str, int] = {}
        intensity_counts: Dict[str, int] = {}
        cross_ref_count = 0
        for t in unique:
            angle_counts[t["angle"]] = angle_counts.get(t["angle"], 0) + 1
            intensity_counts[t["intensity"]] = intensity_counts.get(t["intensity"], 0) + 1
            if t.get("is_cross_reference"):
                cross_ref_count += 1

        metadata = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "total_exported": len(unique),
            "duplicates_removed": len(duplicates),
            "angle_breakdown": angle_counts,
            "intensity_breakdown": intensity_counts,
            "cross_reference_triples": cross_ref_count,
            "format": "conversations",
            "compatible_with": ["huggingface", "unsloth", "axolotl"],
        }

        return jsonl_content, metadata

    def _get_approved_triples(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all approved triples for a session as dicts."""
        with self.db_manager.get_session() as db_session:
            triples = (
                db_session.query(Triple)
                .join(Chunk)
                .join(Document)
                .filter(
                    Document.session_id == session_id,
                    Triple.status == TripleStatus.APPROVED.value,
                )
                .order_by(Chunk.sequence_number, Triple.angle, Triple.intensity)
                .all()
            )
            return [
                {
                    "id": t.id,
                    "system_prompt": t.system_prompt or "",
                    "user_message": t.user_message or "",
                    "assistant_response": t.assistant_response or "",
                    "angle": t.angle,
                    "intensity": t.intensity,
                    "is_cross_reference": t.is_cross_reference,
                }
                for t in triples
            ]

    @staticmethod
    def _detect_duplicates(
        triples: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Detect duplicates by hashing the conversation content.
        Returns (unique_list, duplicate_list).
        """
        seen: set = set()
        unique: List[Dict[str, Any]] = []
        duplicates: List[Dict[str, Any]] = []

        for t in triples:
            fingerprint = hashlib.sha256(
                (t["system_prompt"] + t["user_message"] + t["assistant_response"]).encode()
            ).hexdigest()
            if fingerprint in seen:
                duplicates.append(t)
            else:
                seen.add(fingerprint)
                unique.append(t)

        return unique, duplicates
