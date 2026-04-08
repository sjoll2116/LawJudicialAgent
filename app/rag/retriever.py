"""
混合检索枢纽模块
集成 ChromaDB 向量搜索、Whoosh 全文检索以及知识图谱关系扩展。
"""
from __future__ import annotations

import datetime
import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List

import chromadb
import jieba.analyse
from whoosh.fields import ID, STORED, TEXT, Schema
from whoosh.index import create_in, exists_in, open_dir
from whoosh.qparser import QueryParser
from whoosh.query import And, Or, Term

from app.config import settings
from app.core.logger import get_logger
from app.rag.embedding import SiliconFlowEmbeddingFunction
from app.db.knowledge_graph import KnowledgeGraphManager

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    doc_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


class HybridRetriever:
    _instance: HybridRetriever | None = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        cls._instance = None

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.kg = KnowledgeGraphManager()

        self._embedding_fn = SiliconFlowEmbeddingFunction()
        self._chroma_client = chromadb.PersistentClient(path=settings.seekdb.path)
        self._law_vect = self._chroma_client.get_or_create_collection(
            "law_articles", embedding_function=self._embedding_fn
        )
        self._case_vect = self._chroma_client.get_or_create_collection(
            "court_cases", embedding_function=self._embedding_fn
        )

        self._schema = Schema(
            id=ID(stored=True, unique=True),
            content=TEXT(stored=True, analyzer=jieba.analyse.ChineseAnalyzer()),
            role=ID(stored=True),
            logic_type=ID(stored=True),
            domain=ID(stored=True),
            case_id=ID(stored=True),
            case_name=ID(stored=True),
            law_name=ID(stored=True),
            document_name=ID(stored=True),
            doc_subtype=ID(stored=True),
            elements=TEXT(stored=True, analyzer=jieba.analyse.ChineseAnalyzer()),
            keywords=TEXT(stored=True, analyzer=jieba.analyse.ChineseAnalyzer()),
            metadata=STORED(),
        )
        self._fts_lock = threading.Lock()
        self._init_fts()

    def get_available_metadata(self) -> Dict[str, List[str]]:
        path = os.path.join(settings.seekdb.path, "../../data/metadata_registry.json")
        path = os.path.normpath(path)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"elements": [], "keywords": []}

    def _init_fts(self):
        idx_dir = settings.system.whoosh_path
        os.makedirs(idx_dir, exist_ok=True)

        if exists_in(idx_dir):
            self._fts_index = open_dir(idx_dir)
            required_fields = {"elements", "doc_subtype", "domain"}
            if not required_fields.issubset(set(self._fts_index.schema.names())):
                logger.warning("Whoosh schema mismatch detected. Recreating index...")
                self._fts_index = create_in(idx_dir, self._schema)
        else:
            self._fts_index = create_in(idx_dir, self._schema)

    def _normalize_filter_key(self, key: str) -> str:
        if key == "section_type":
            return "logic_type"
        return key

    def _fts_search(
        self,
        collection_name: str,
        query_text: str,
        where: Dict | None = None,
        n_results: int = 10,
    ) -> List[RetrievalResult]:
        del collection_name  # reserved for future split-index support
        results: list[RetrievalResult] = []

        with self._fts_index.searcher() as searcher:
            parser = QueryParser("content", self._fts_index.schema)
            query = parser.parse(query_text)

            filters = []
            relevant_keys = {
                "logic_type",
                "domain",
                "case_id",
                "law_name",
                "role",
                "elements",
                "keywords",
                "doc_subtype",
                "document_name",
            }

            def parse_condition(key: str, value: Any):
                key = self._normalize_filter_key(key)
                if key not in relevant_keys:
                    return None
                if isinstance(value, dict) and "$eq" in value:
                    value = value["$eq"]
                if isinstance(value, list):
                    return Or([Term(key, item) for item in value])
                return Term(key, value)

            if where:
                if "$and" in where:
                    for cond in where["$and"]:
                        for k, v in cond.items():
                            term = parse_condition(k, v)
                            if term is not None:
                                filters.append(term)
                else:
                    for k, v in where.items():
                        term = parse_condition(k, v)
                        if term is not None:
                            filters.append(term)

            if filters:
                query = And([query] + filters)

            hits = searcher.search(query, limit=n_results)
            for hit in hits:
                results.append(
                    RetrievalResult(
                        doc_id=hit["id"],
                        content=hit["content"],
                        metadata=hit["metadata"],
                        score=float(hit.score),
                    )
                )

        return results

    def _rrf_combine(
        self,
        v_res: List[RetrievalResult],
        f_res: List[RetrievalResult],
        w_v: float = 1.0,
        w_f: float = 1.0,
        k: int = 60,
    ) -> List[RetrievalResult]:
        scores: dict[str, float] = {}
        result_map: dict[str, RetrievalResult] = {}

        for rank, item in enumerate(v_res):
            scores[item.doc_id] = scores.get(item.doc_id, 0.0) + w_v * (1.0 / (k + rank + 1))
            result_map[item.doc_id] = item

        for rank, item in enumerate(f_res):
            scores[item.doc_id] = scores.get(item.doc_id, 0.0) + w_f * (1.0 / (k + rank + 1))
            if item.doc_id not in result_map:
                result_map[item.doc_id] = item

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [
            RetrievalResult(
                doc_id=doc_id,
                content=result_map[doc_id].content,
                metadata=result_map[doc_id].metadata,
                score=scores[doc_id],
            )
            for doc_id in sorted_ids
        ]

    def _complete_case_context(self, cases: list[RetrievalResult]) -> list[RetrievalResult]:
        completed: list[RetrievalResult] = []
        seen: set[str] = set()

        for item in cases:
            case_id = item.metadata.get("case_id")
            logic_type = item.metadata.get("logic_type")
            dedup_key = f"{case_id}_{logic_type}"

            if not case_id or dedup_key in seen:
                if not case_id:
                    completed.append(item)
                continue

            try:
                parts = self._case_vect.get(where={"case_id": {"$eq": case_id}, "logic_type": {"$eq": logic_type}})
                if parts and parts["documents"]:
                    merged = "\n".join(
                        d
                        for _, d in sorted(
                            zip(parts["metadatas"], parts["documents"]),
                            key=lambda x: x[0].get("part_idx", 0),
                        )
                    )
                    item.content = merged
                    completed.append(item)
                    seen.add(dedup_key)
                else:
                    completed.append(item)
            except Exception:
                completed.append(item)

        return completed

    def _apply_legal_boost(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        current_year = datetime.datetime.now().year
        for item in results:
            boost = 1.0
            court_level = item.metadata.get("court_level")
            if court_level == 1:
                boost *= 1.2
            elif court_level == 2:
                boost *= 1.1

            year = int(item.metadata.get("year", 0) or 0)
            if year > 0:
                if current_year - year <= 2:
                    boost *= 1.1
                elif current_year - year <= 5:
                    boost *= 1.05

            item.score *= boost

        return sorted(results, key=lambda x: x.score, reverse=True)

    def _phase_role_filters(self, phase: str | None, user_role: str | None) -> dict[str, list[str]]:
        logic_types: list[str] = []

        if phase in {"evidence_exchange", "evidence"}:
            logic_types.extend(["attack", "counter", "fact"])
        elif phase in {"conclusion", "judgment", "trial"}:
            logic_types.extend(["reasoning", "final_order", "verdict", "fact"])
        elif phase in {"document"}:
            logic_types.extend(["reasoning", "final_order", "verdict"])

        if user_role == "plaintiff_side":
            logic_types.extend(["attack", "fact"])
        elif user_role == "defendant_side":
            logic_types.extend(["counter", "fact"])

        if not logic_types:
            return {}

        # 去重
        deduped = list(dict.fromkeys(logic_types))
        return {"logic_type": deduped}

    def search_all(
        self,
        query_text: str,
        domain: str | None = None,
        n_results: int = 5,
        filters: Dict | None = None,
        user_role: str | None = None,
        phase: str | None = None,
        **kwargs,
    ) -> dict:
        del kwargs

        merged_filters: dict[str, Any] = {}
        if filters:
            merged_filters.update(filters)
        merged_filters.update(self._phase_role_filters(phase, user_role))

        law_where = dict(merged_filters)
        case_where = dict(merged_filters)

        if domain:
            law_where["domain"] = {"$eq": domain}
            case_where["domain"] = {"$eq": domain}

        law_res = self.search_law_articles(q=query_text, where=law_where or None, n=n_results)
        case_res = self.search_court_cases(q=query_text, where=case_where or None, n=n_results)

        return {
            "law_articles": law_res,
            "court_cases": self._apply_legal_boost(self._complete_case_context(case_res)),
        }

    def format_context_for_prompt(self, res: dict) -> str:
        prompt = "### 法律依据库 (Structured Legal Registry)\n"
        
        # 结果分类
        direct_laws = []
        related_interpretations = []
        
        for item in res.get("law_articles", []):
            is_related = item.metadata.get("is_related", False)
            if item.metadata.get("doc_subtype") == "interpretation" or is_related:
                related_interpretations.append(item)
            else:
                direct_laws.append(item)

        if direct_laws:
            prompt += "\n#### 【第一层：直接法律依据 (Direct Authorities)】\n"
            prompt += "> 此类为法律原条文，具有最高适用位阶。\n"
            for item in direct_laws:
                source_id = item.doc_id
                prompt += (
                    f"- [ID={source_id}] 《{item.metadata.get('law_name', '未知法条')}》第{item.metadata.get('article_num', '-')}条\n"
                    f"  内容: {item.content.replace('[裁判依据: ', '').split(']')[-1].strip()}\n"
                )

        if related_interpretations:
            prompt += "\n#### 【第二层：关联司法解释/细则 (Related Interpretations)】\n"
            prompt += "> 此类是对上述法律条文的具体适用说明，具有极高裁判参考价值。\n"
            for item in related_interpretations:
                source_id = item.doc_id
                is_related = item.metadata.get("is_related", False)
                rel_hint = " [系统关联匹配]" if is_related else ""
                prompt += (
                    f"- [ID={source_id}]{rel_hint} 《{item.metadata.get('law_name', '未知解释')}》\n"
                    f"  内容: {item.content.replace('[裁判依据: ', '').split(']')[-1].strip()}\n"
                )

        if res.get("court_cases"):
            prompt += "\n#### 【第三层：相关裁判案例参考 (Judicial Precedents)】\n"
            prompt += "> 仅用于理解事实认定逻辑，不建议直接作为法律依据引用。\n"
            for item in res["court_cases"]:
                header = item.metadata.get("header_context", "")
                full_content = f"{header}{item.content}" if header else item.content
                source_id = item.doc_id
                prompt += (
                    f"**[ID={source_id}] [{item.metadata.get('case_name', '未知案例')}]** "
                    f"({item.metadata.get('case_id', '-')})\n{full_content}\n\n---\n"
                )

        return prompt

    def build_source_catalog(self, res: dict) -> list[dict[str, Any]]:
        catalog: list[dict[str, Any]] = []
        for item in res.get("law_articles", []):
            catalog.append(
                {
                    "source_id": item.doc_id,
                    "source_type": "law_article",
                    "label": item.metadata.get("law_name", "未知法条"),
                    "logic_type": item.metadata.get("logic_type", "law_article"),
                    "summary": item.metadata.get("summary", ""),
                    "law_name": item.metadata.get("law_name", ""),
                    "case_id": "",
                }
            )
        for item in res.get("court_cases", []):
            catalog.append(
                {
                    "source_id": item.doc_id,
                    "source_type": "court_case",
                    "label": item.metadata.get("case_name", "未知案例"),
                    "logic_type": item.metadata.get("logic_type", ""),
                    "summary": item.metadata.get("summary", ""),
                    "law_name": "",
                    "case_id": item.metadata.get("case_id", ""),
                }
            )
        return catalog

    def search_law_articles(
        self,
        q: str | None = None,
        where: Dict | None = None,
        n: int = 5,
        query_text: str | None = None,
        n_results: int | None = None,
    ):
        q = query_text or q or ""
        n = n_results or n

        vect = self._law_vect.query(query_texts=[q], where=where, n_results=n)
        v_res = [
            RetrievalResult(doc_id=i, content=c, metadata=m)
            for i, c, m in zip(vect["ids"][0], vect["documents"][0], vect["metadatas"][0])
        ]
        f_res = self._fts_search("law_articles", q, where, n)

        alpha = float(max(0.0, min(1.0, settings.system.law_weight_alpha)))
        initial_hits = self._rrf_combine(v_res, f_res, w_v=alpha, w_f=1.0 - alpha)
        
        # --- 关系扩展 (Knowledge Graph Extension) ---
        expanded_hits = list(initial_hits)
        seen_ids = {h.doc_id for h in initial_hits}
        
        for hit in initial_hits:
            prov_id = hit.metadata.get("provision_id")
            if prov_id:
                related = self.kg.get_related_provisions(prov_id)
                for rel in related:
                    # 尝试从向量库或重新构建一个 RetrievalResult
                    # 这里简化为：如果它已经在 initial_hits 里了，跳过；否则标记为 is_related 加入
                    doc_id = f"graph_{rel['id']}" # 假设
                    if doc_id not in seen_ids:
                        expanded_hits.append(RetrievalResult(
                            doc_id=doc_id,
                            content=rel['content'],
                            metadata={
                                "law_name": rel['doc_title'],
                                "doc_subtype": rel['doc_type'].lower(),
                                "is_related": True,
                                "article_num": rel['provision_no']
                            },
                            score=hit.score * 0.9 # 给关联结果稍低的权重，但依然保留
                        ))
                        seen_ids.add(doc_id)
        
        return expanded_hits

    def search_court_cases(
        self,
        q: str | None = None,
        where: Dict | None = None,
        n: int = 5,
        query_text: str | None = None,
        n_results: int | None = None,
    ):
        q = query_text or q or ""
        n = n_results or n

        simple_where = {self._normalize_filter_key(k): v for k, v in (where or {}).items() if not isinstance(v, list)}
        vect = self._case_vect.query(query_texts=[q], where=simple_where or None, n_results=n)
        v_res = [
            RetrievalResult(doc_id=i, content=c, metadata=m)
            for i, c, m in zip(vect["ids"][0], vect["documents"][0], vect["metadatas"][0])
        ]
        f_res = self._fts_search("court_cases", q, where, n)

        alpha = float(max(0.0, min(1.0, settings.system.case_weight_alpha)))
        return self._rrf_combine(v_res, f_res, w_v=alpha, w_f=1.0 - alpha)

    def ingest_law_articles(self, ids: List[str], documents: List[str], metadatas: List[Dict]):
        self._law_vect.upsert(ids=ids, documents=documents, metadatas=metadatas)
        try:
            with self._fts_lock:
                writer = self._fts_index.writer()
                for doc_id, doc, meta in zip(ids, documents, metadatas):
                    writer.update_document(
                        id=doc_id,
                        content=doc,
                        role="neutral_court",
                        logic_type="law_article",
                        domain=meta.get("domain", "general"),
                        law_name=meta.get("law_name"),
                        document_name=meta.get("document_name", meta.get("law_name")),
                        doc_subtype=meta.get("doc_subtype", "law"),
                        metadata=meta,
                    )
                writer.commit()
        except Exception as e:
            self._law_vect.delete(ids=ids)
            raise e

    def ingest_court_cases(self, ids: List[str], documents: List[str], metadatas: List[Dict]):
        self._case_vect.upsert(ids=ids, documents=documents, metadatas=metadatas)
        try:
            with self._fts_lock:
                writer = self._fts_index.writer()
                for doc_id, doc, meta in zip(ids, documents, metadatas):
                    elements = " ".join(meta.get("elements", [])) if isinstance(meta.get("elements"), list) else meta.get("elements", "")
                    keywords = " ".join(meta.get("keywords", [])) if isinstance(meta.get("keywords"), list) else meta.get("keywords", "")
                    writer.update_document(
                        id=doc_id,
                        content=doc,
                        role=meta.get("role", "neutral_court"),
                        logic_type=meta.get("logic_type", "court_case"),
                        domain=meta.get("domain", "general"),
                        case_id=meta.get("case_id"),
                        case_name=meta.get("case_name"),
                        elements=elements,
                        keywords=keywords,
                        metadata=meta,
                    )
                writer.commit()
        except Exception as e:
            self._case_vect.delete(ids=ids)
            raise e

    def delete_file_by_name(self, dtype: str, fname: str):
        try:
            coll = self._case_vect if dtype == "case" else self._law_vect
            if dtype == "case":
                coll.delete(where={"case_name": {"$eq": fname}})
            elif dtype == "interpretation":
                coll.delete(where={"$and": [{"document_name": {"$eq": fname}}, {"doc_subtype": {"$eq": "interpretation"}}]})
            else:
                coll.delete(where={"$and": [{"document_name": {"$eq": fname}}, {"doc_subtype": {"$eq": "law"}}]})
                coll.delete(where={"law_name": {"$eq": fname}})

            with self._fts_lock:
                writer = self._fts_index.writer()
                if dtype == "case":
                    writer.delete_by_term("case_name", fname)
                elif dtype == "interpretation":
                    writer.delete_by_query(And([Term("document_name", fname), Term("doc_subtype", "interpretation")]))
                else:
                    writer.delete_by_query(And([Term("document_name", fname), Term("doc_subtype", "law")]))
                    writer.delete_by_term("law_name", fname)
                writer.commit()

            import glob

            pattern = os.path.join("temp_uploads", f"{fname}*")
            for file_path in glob.glob(pattern):
                try:
                    os.remove(file_path)
                    logger.info("Physical file permanently deleted: %s", file_path)
                except Exception as e:
                    logger.error("Failed to remove physical file %s: %s", file_path, e)

            return True
        except Exception as e:
            logger.error("Full deletion failed for %s: %s", fname, e)
            return False
