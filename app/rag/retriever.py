from __future__ import annotations
import logging, os, json, datetime, chromadb, jieba.analyse, threading
from dataclasses import dataclass, field
from typing import Any, List, Dict, Union, Optional
from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.qparser import QueryParser
from whoosh.query import Term, And
from app.config import settings
from app.rag.embedding import SiliconFlowEmbeddingFunction

from app.core.logger import get_logger
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
        if cls._instance is None: cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """强制销毁单例，用于热重载对齐"""
        cls._instance = None

    def __init__(self):
        if hasattr(self, "_initialized"): return
        self._initialized = True
        self._embedding_fn = SiliconFlowEmbeddingFunction()
        self._chroma_client = chromadb.PersistentClient(path=settings.seekdb.path)
        self._law_vect = self._chroma_client.get_or_create_collection("law_articles", embedding_function=self._embedding_fn)
        self._case_vect = self._chroma_client.get_or_create_collection("court_cases", embedding_function=self._embedding_fn)
        self._schema = Schema(
            id=ID(stored=True, unique=True),
            content=TEXT(stored=True, analyzer=jieba.analyse.ChineseAnalyzer()),
            role=ID(stored=True),
            logic_type=ID(stored=True),
            case_id=ID(stored=True),
            case_name=ID(stored=True),
            law_name=ID(stored=True),
            document_name=ID(stored=True),
            doc_subtype=ID(stored=True),
            elements=TEXT(stored=True, analyzer=jieba.analyse.ChineseAnalyzer()),
            keywords=TEXT(stored=True, analyzer=jieba.analyse.ChineseAnalyzer()),
            metadata=STORED()
        )
        self._fts_lock = threading.Lock()
        self._init_fts()

    def get_available_metadata(self) -> Dict[str, List[str]]:
        """获取全库已注册的可用要素与关键词"""
        path = os.path.join(settings.seekdb.path, "../../data/metadata_registry.json")
        path = os.path.normpath(path)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {"elements": [], "keywords": []}

    def _init_fts(self):
        idx_dir = settings.system.whoosh_path
        os.makedirs(idx_dir, exist_ok=True)
        
        if exists_in(idx_dir):
            self._fts_index = open_dir(idx_dir)
            # 校验物理索引 Schema 是否包含最新字段 (如 elements)
            # 如果不包含，说明是旧版遗留索引，必须粉碎重建
            if "elements" not in self._fts_index.schema or "doc_subtype" not in self._fts_index.schema:
                logger.warning("Whoosh schema mismatch detected. Recreating index...")
                # 由于 Windows 下可能有文件占用，这里建议手动或程序尝试清理
                # 我们采用最稳健的逻辑：如果 open 后的 schema 不对，就重新 create
                self._fts_index = create_in(idx_dir, self._schema)
        else:
            self._fts_index = create_in(idx_dir, self._schema)

    def _fts_search(self, collection_name: str, query_text: str, where: Dict = None, n_results: int = 10) -> List[RetrievalResult]:
        res = []
        with self._fts_index.searcher() as s:
            parser = QueryParser("content", self._fts_index.schema)
            # 基础文本解析
            q = parser.parse(query_text)
            
            filters = []
            if where:
                # 处理 AND/OR 嵌套 (简化支持)
                relevant_keys = [
                    "logic_type", "domain", "case_id", "law_name", "role",
                    "elements", "keywords", "doc_subtype", "document_name"
                ]
                
                def parse_condition(k, v):
                    if isinstance(v, dict) and "$eq" in v: v = v["$eq"]
                    if isinstance(v, list):
                        from whoosh.query import Or
                        return Or([Term(k, item) for item in v])
                    return Term(k, v)

                # 手动处理 level 1 $and
                if "$and" in where:
                    for cond in where["$and"]:
                        for k, v in cond.items():
                            if k in relevant_keys: filters.append(parse_condition(k, v))
                else:
                    for k, v in where.items():
                        if k in relevant_keys: filters.append(parse_condition(k, v))
            
            if filters: q = And([q] + filters)
            hits = s.search(q, limit=n_results)
            for h in hits: res.append(RetrievalResult(doc_id=h["id"], content=h["content"], metadata=h["metadata"], score=h.score))
        return res

    def _rrf_combine(self, v_res: List[RetrievalResult], f_res: List[RetrievalResult], w_v: float = 1.0, w_f: float = 1.0, k: int = 60) -> List[RetrievalResult]:
        scores, d_map = {}, {}
        for r, d in enumerate(v_res):
            scores[d.doc_id] = scores.get(d.doc_id, 0) + w_v * (1.0 / (k + r + 1))
            d_map[d.doc_id] = d
        for r, d in enumerate(f_res):
            scores[d.doc_id] = scores.get(d.doc_id, 0) + w_f * (1.0 / (k + r + 1))
            if d.doc_id not in d_map: d_map[d.doc_id] = d
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [RetrievalResult(doc_id=i, content=d_map[i].content, metadata=d_map[i].metadata, score=scores[i]) for i in sorted_ids]

    def _complete_case_context(self, cases: list[RetrievalResult]) -> list[RetrievalResult]:
        comp, seen = [], set()
        for r in cases:
            cid, ltype = r.metadata.get("case_id"), r.metadata.get("logic_type")
            if not cid or f"{cid}_{ltype}" in seen:
                if not cid: comp.append(r)
                continue
            try:
                ps = self._case_vect.get(where={"case_id": {"$eq": cid}, "logic_type": {"$eq": ltype}})
                if ps and ps["documents"]:
                    # 合并具有相同逻辑块的所有切片
                    txt = "\n".join([d for _, d in sorted(zip(ps["metadatas"], ps["documents"]), key=lambda x: x[0].get("part_idx", 0))])
                    r.content, comp = txt, comp + [r]
                    seen.add(f"{cid}_{ltype}")
                else: comp.append(r)
            except: comp.append(r)
        return comp

    def _apply_legal_boost(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        cur_y = datetime.datetime.now().year
        for r in results:
            b = (1.2 if r.metadata.get("court_level") == 1 else 1.1 if r.metadata.get("court_level") == 2 else 1.0)
            y = int(r.metadata.get("year", 0))
            if y > 0: b *= (1.1 if cur_y - y <= 2 else 1.05 if cur_y - y <= 5 else 1.0)
            r.score *= b
        return sorted(results, key=lambda x: x.score, reverse=True)

    def search_all(self, query_text: str, domain: str = "外商投资", n_results: int = 5, filters: Dict = None) -> dict:
        """
        全量检索聚合器：同时从法条库和案例库召回数据。
        不再进行硬编码的业务拦截，由调用方显式决定过滤逻辑。
        """
        # 1. 检索法条库 (Law Collection)
        law_res = self.search_law_articles(query_text, {"domain": {"$eq": domain}}, n_results)
        
        # 2. 构造案例库过滤条件 (Case Collection)
        c_filter = {"domain": {"$eq": domain}}
        
        # 注入显式过滤参数 (如 logic_type)
        if filters:
            for k in ["elements", "keywords", "logic_type", "section_type"]:
                if filters.get(k):
                    target_key = "logic_type" if k == "section_type" else k
                    c_filter[target_key] = filters[k]

        # 3. 检索案例库
        cases = self.search_court_cases(query_text, c_filter, n_results)
        return {
            "law_articles": law_res, 
            "court_cases": self._apply_legal_boost(self._complete_case_context(cases))
        }

    def format_context_for_prompt(self, res: dict) -> str:
        p = "### 法律依据库\n"
        if res.get("law_articles"):
            p += "\n#### 1. 裁判依据 (法条)\n"
            for r in res["law_articles"]: p += f"- 《{r.metadata.get('law_name')}》: {r.content}\n"
        if res.get("court_cases"):
            p += "\n#### 2. 参考判例\n"
            for r in res["court_cases"]:
                # 重建实时上下文：从元数据中还原“标签+背景+板块摘要”
                header = r.metadata.get("header_context", "")
                full_content = header + r.content if header else r.content
                p += f"**[{r.metadata.get('case_name')}]** ({r.metadata.get('case_id')})\n{full_content}\n\n---\n"
        return p

    def search_law_articles(self, q: str, where: Dict = None, n: int = 5):
        v = self._law_vect.query(query_texts=[q], where=where, n_results=n)
        vr = [RetrievalResult(doc_id=i, content=c, metadata=m) for i, c, m in zip(v["ids"][0], v["documents"][0], v["metadatas"][0])]
        fr = self._fts_search("law_articles", q, where, n)
        return self._rrf_combine(vr, fr, w_v=0.2, w_f=0.8)

    def search_court_cases(self, q: str, where: Dict = None, n: int = 5):
        # 向量检索尽量保留简单过滤
        simple_where = {k: v for k, v in where.items() if not isinstance(v, list)} if where else None
        v = self._case_vect.query(query_texts=[q], where=simple_where, n_results=n)
        vr = [RetrievalResult(doc_id=i, content=c, metadata=m) for i, c, m in zip(v["ids"][0], v["documents"][0], v["metadatas"][0])]
        
        # FTS 检索负责处理复杂的 $or / list 过滤
        fr = self._fts_search("court_cases", q, where, n)
        return self._rrf_combine(vr, fr, w_v=0.7, w_f=0.3)

    def ingest_law_articles(self, ids: List[str], documents: List[str], metadatas: List[Dict]):
        self._law_vect.add(ids=ids, documents=documents, metadatas=metadatas)
        try:
            with self._fts_lock:
                writer = self._fts_index.writer()
                for i, d, m in zip(ids, documents, metadatas):
                    writer.update_document(
                        id=i,
                        content=d,
                        role="neutral_court",
                        logic_type="law_article",
                        law_name=m.get("law_name"),
                        document_name=m.get("document_name", m.get("law_name")),
                        doc_subtype=m.get("doc_subtype", "law"),
                        metadata=m
                    )
                writer.commit()
        except Exception as e:
            self._law_vect.delete(ids=ids)
            raise e

    def ingest_court_cases(self, ids: List[str], documents: List[str], metadatas: List[Dict]):
        self._case_vect.add(ids=ids, documents=documents, metadatas=metadatas)
        try:
            with self._fts_lock:
                writer = self._fts_index.writer()
                for i, d, m in zip(ids, documents, metadatas):
                    # 将列表转换为带分隔符的字符串以便 Whoosh 检索
                    els = " ".join(m.get("elements", [])) if isinstance(m.get("elements"), list) else m.get("elements", "")
                    kws = " ".join(m.get("keywords", [])) if isinstance(m.get("keywords"), list) else m.get("keywords", "")
                    writer.update_document(id=i, content=d, role=m.get("role", "neutral_court"), logic_type=m.get("logic_type", "court_case"),
                                          case_id=m.get("case_id"), case_name=m.get("case_name"), 
                                          elements=els, keywords=kws, metadata=m)
                writer.commit()
        except Exception as e:
            self._case_vect.delete(ids=ids)
            raise e

    def delete_file_by_name(self, dtype: str, fname: str):
        """
        全流程销毁：ChromaDB + Whoosh + 磁盘物理文件
        """
        try:
            # 1. ChromaDB 逻辑删除
            coll = self._case_vect if dtype == "case" else self._law_vect
            if dtype == "case":
                coll.delete(where={"case_name": {"$eq": fname}})
            elif dtype == "interpretation":
                coll.delete(
                    where={
                        "$and": [
                            {"document_name": {"$eq": fname}},
                            {"doc_subtype": {"$eq": "interpretation"}}
                        ]
                    }
                )
            else:
                # Prefer subtype deletion first, then fallback for legacy rows.
                coll.delete(
                    where={
                        "$and": [
                            {"document_name": {"$eq": fname}},
                            {"doc_subtype": {"$eq": "law"}}
                        ]
                    }
                )
                coll.delete(where={"law_name": {"$eq": fname}})

            # 2. Whoosh 索引抹除 (带锁)
            with self._fts_lock:
                w = self._fts_index.writer()
                if dtype == "case":
                    w.delete_by_term("case_name", fname)
                elif dtype == "interpretation":
                    w.delete_by_query(And([Term("document_name", fname), Term("doc_subtype", "interpretation")]))
                else:
                    w.delete_by_query(And([Term("document_name", fname), Term("doc_subtype", "law")]))
                    w.delete_by_term("law_name", fname)
                w.commit()

            # 3. 磁盘物理文件销毁 (改进：支持多后缀模糊匹配，彻底清除存档)
            import glob
            # 匹配原文件 (docx/pdf/txt) 以及处理过程中可能产生的同名清洗版本
            pattern = os.path.join("temp_uploads", f"{fname}*")
            matches = glob.glob(pattern)
            
            for file_path in matches:
                try:
                    os.remove(file_path)
                    logger.info(f"Physical file permanently deleted: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to remove physical file {file_path}: {e}")
                
            return True
        except Exception as e:
            logger.error(f"Full deletion failed for {fname}: {e}")
            return False
