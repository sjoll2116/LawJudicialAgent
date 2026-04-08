import json
import os
import sys
from pathlib import Path
"""
后端业务逻辑层，负责协调 RAG 检索、Agent 编排与数据库操作。
"""
from typing import Any, Dict, List

# Ensure app is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import pythoncom
except ImportError:
    pythoncom = None

from langchain_core.messages import HumanMessage

from app.config import PROJECT_ROOT
from app.core.logger import get_logger
from app.graph.state import CaseState, Citation, Claim, PartyInfo
from app.graph.workflow import app_graph
from app.rag.law_processor import LawDataProcessor
from app.rag.retriever import HybridRetriever

logger = get_logger(__name__)


class LegalApiService:
    def __init__(self):
        HybridRetriever.reset_instance()
        self.retriever = HybridRetriever()
        self.processor = LawDataProcessor()
        self.history_file = PROJECT_ROOT / "data" / "ingest_history.json"
        self.relation_file = PROJECT_ROOT / "data" / "relation_registry.json"
        self.ingestion_tasks: Dict[str, Any] = {}

    def _export_state(self, state: CaseState) -> Dict[str, Any]:
        payload = state.model_dump()
        payload["messages"] = state.messages
        payload["intent"] = state.intent.value if hasattr(state.intent, "value") else state.intent
        return payload

    def _coerce_result_types(self, state: CaseState, result: dict) -> None:
        for key, val in result.items():
            if key == "messages":
                state.messages = val
                continue
            if not hasattr(state, key):
                continue

            if key == "parties" and isinstance(val, dict):
                coerced = {}
                for party_id, party_val in val.items():
                    try:
                        coerced[party_id] = (
                            party_val
                            if isinstance(party_val, PartyInfo)
                            else PartyInfo.model_validate(party_val)
                        )
                    except Exception:
                        continue
                setattr(state, key, coerced)
                continue

            if key == "claims" and isinstance(val, list):
                coerced_claims = []
                for claim_val in val:
                    try:
                        coerced_claims.append(
                            claim_val if isinstance(claim_val, Claim) else Claim.model_validate(claim_val)
                        )
                    except Exception:
                        continue
                setattr(state, key, coerced_claims)
                continue

            if key == "judgment_citations" and isinstance(val, list):
                coerced_citations = []
                for citation_val in val:
                    try:
                        coerced_citations.append(
                            citation_val
                            if isinstance(citation_val, Citation)
                            else Citation.model_validate(citation_val)
                        )
                    except Exception:
                        continue
                setattr(state, key, coerced_citations)
                continue

            setattr(state, key, val)

    def chat(self, user_message: str, state_dict: Dict[str, Any] = None) -> Dict[str, Any]:
        state = CaseState(**state_dict) if state_dict else CaseState()
        state.messages.append(HumanMessage(content=user_message))
        result = app_graph.invoke(state.model_dump())

        if isinstance(result, dict):
            self._coerce_result_types(state, result)

        return self._export_state(state)

    def _persist_relation_registry(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            return

        first_meta = chunks[0].get("metadata", {})
        if first_meta.get("doc_subtype") != "interpretation":
            return

        document_name = first_meta.get("document_name") or first_meta.get("law_name")
        if not document_name:
            return

        related_laws = sorted(
            {
                law
                for c in chunks
                for law in c.get("metadata", {}).get("related_laws", [])
                if law
            }
        )
        related_articles = sorted(
            {
                article
                for c in chunks
                for article in c.get("metadata", {}).get("related_articles", [])
                if article
            }
        )

        os.makedirs(self.relation_file.parent, exist_ok=True)
        payload: Dict[str, Any] = {}
        if self.relation_file.exists():
            try:
                with open(self.relation_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                payload = {}

        payload[document_name] = {
            "doc_subtype": "interpretation",
            "related_laws": related_laws,
            "related_articles": related_articles,
        }
        with open(self.relation_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def get_all_chunks(self) -> Dict[str, List[Any]]:
        law_data = self.retriever._law_vect.get(limit=9999)
        case_data = self.retriever._case_vect.get(limit=9999)
        return {
            "law_articles": [
                {"id": i, "content": d, "metadata": m}
                for i, d, m in zip(law_data["ids"], law_data["documents"], law_data["metadatas"])
            ]
            if law_data
            else [],
            "court_cases": [
                {"id": i, "content": d, "metadata": m}
                for i, d, m in zip(case_data["ids"], case_data["documents"], case_data["metadatas"])
            ]
            if case_data
            else [],
        }

    def delete_chunk(self, collection_type: str, chunk_id: str):
        if collection_type == "law":
            self.retriever._law_vect.delete(ids=[chunk_id])
        else:
            self.retriever._case_vect.delete(ids=[chunk_id])
        return True

    def get_ingestion_tasks(self) -> Dict[str, Any]:
        return self.ingestion_tasks

    async def ingest_file(self, file_path: Path, doc_type: str = None):
        file_name = file_path.name
        self.ingestion_tasks[file_name] = {"status": "processing", "error": None}

        if pythoncom:
            pythoncom.CoInitialize()

        try:
            from scripts.ingest_legal_data import read_text_from_file

            content = read_text_from_file(file_path)
            stem_name = file_path.stem

            normalized_doc_type = (doc_type or "").lower().strip()
            is_case = (normalized_doc_type == "case") or (
                not normalized_doc_type and ("court_cases" in str(file_path) or "case" in stem_name.lower())
            )

            if is_case:
                chunks = await self.processor.process_court_case(stem_name, content)
                if not chunks:
                    raise ValueError(f"Failed to generate chunks for case: {stem_name}")
                
                self.retriever.ingest_court_cases(
                    ids=[c["id"] for c in chunks],
                    documents=[c["content"] for c in chunks],
                    metadatas=[c["metadata"] for c in chunks],
                )
            else:
                inferred_subtype = normalized_doc_type
                if inferred_subtype not in {"law", "interpretation"}:
                    inferred_subtype = "interpretation" if any(
                        marker in stem_name for marker in ["司法解释", "解释"]
                    ) else "law"

                chunks = self.processor.process_law_article(
                    stem_name,
                    content,
                    doc_subtype=inferred_subtype,
                )
                if not chunks:
                    raise ValueError(f"Failed to generate chunks for law: {stem_name}")
                
                self.retriever.ingest_law_articles(
                    ids=[c["id"] for c in chunks],
                    documents=[c["content"] for c in chunks],
                    metadatas=[c["metadata"] for c in chunks],
                )
                self._persist_relation_registry(chunks)

            self.ingestion_tasks[file_name] = {"status": "completed", "error": None}
            return True

        except Exception as e:
            err_msg = str(e)
            logger.error("Ingestion error for %s: %s", file_name, err_msg, exc_info=True)
            self.ingestion_tasks[file_name] = {"status": "failed", "error": err_msg}
            raise e
        finally:
            if pythoncom:
                pythoncom.CoUninitialize()
