import sqlite3
import json
import os
from typing import List, Dict, Any, Optional
from app.config import PROJECT_ROOT

class KnowledgeGraphManager:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(PROJECT_ROOT / "data" / "knowledge_graph.sqlite")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            # Document Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    normalized_title TEXT,
                    doc_type TEXT NOT NULL, -- LAW / INTERPRETATION
                    doc_no TEXT,
                    status TEXT DEFAULT 'EFFECTIVE',
                    authority_level INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Provision Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS provision (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    provision_no TEXT, -- 第一条
                    content TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES document(id)
                )
            """)
            # Relation Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL, -- EXPLAINS_PROVISION / CITES / APPLIES_TO
                    confidence REAL DEFAULT 1.0,
                    evidence_text TEXT
                )
            """)
            # Topic Table (Simplified for start)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS topic (
                    name TEXT PRIMARY KEY,
                    description TEXT
                )
            """)
            # Provision-Topic Link
            conn.execute("""
                CREATE TABLE IF NOT EXISTS provision_topic (
                    provision_id TEXT,
                    topic_name TEXT,
                    PRIMARY KEY (provision_id, topic_name)
                )
            """)
            conn.commit()

    def add_document(self, doc_id: str, title: str, doc_type: str, doc_no: str = None, authority_level: int = 3):
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO document (id, title, doc_type, doc_no, authority_level)
                VALUES (?, ?, ?, ?, ?)
            """, (doc_id, title, doc_type, doc_no, authority_level))
            conn.commit()

    def add_provision(self, prov_id: str, doc_id: str, provision_no: str, content: str):
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO provision (id, document_id, provision_no, content)
                VALUES (?, ?, ?, ?)
            """, (prov_id, doc_id, provision_no, content))
            conn.commit()

    def add_relation(self, source_id: str, target_id: str, rel_type: str, confidence: float = 1.0, evidence: str = None):
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO relation (source_id, target_id, relation_type, confidence, evidence_text)
                VALUES (?, ?, ?, ?, ?)
            """, (source_id, target_id, rel_type, confidence, evidence))
            conn.commit()

    def get_related_provisions(self, provision_id: str) -> List[Dict[str, Any]]:
        """Find associated provisions (One-hop expansion)"""
        results = []
        with self._get_conn() as conn:
            # 1. Find interpretations for this law article
            rows = conn.execute("""
                SELECT p.*, d.title as doc_title, d.doc_type 
                FROM provision p
                JOIN relation r ON p.id = r.source_id
                JOIN document d ON p.document_id = d.id
                WHERE r.target_id = ? AND r.relation_type = 'EXPLAINS_PROVISION'
            """, (provision_id,)).fetchall()
            for row in rows:
                results.append(dict(row))
            
            # 2. Find laws that this interpretation cites
            rows = conn.execute("""
                SELECT p.*, d.title as doc_title, d.doc_type 
                FROM provision p
                JOIN relation r ON p.id = r.target_id
                JOIN document d ON p.document_id = d.id
                WHERE r.source_id = ? AND r.relation_type = 'EXPLAINS_PROVISION'
            """, (provision_id,)).fetchall()
            for row in rows:
                results.append(dict(row))
                
        return results

    def get_document_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM document WHERE title = ? OR normalized_title = ?", (title, title)).fetchone()
            return dict(row) if row else None
