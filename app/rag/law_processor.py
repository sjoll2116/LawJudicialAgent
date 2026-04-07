"""
法条与文书的数据清洗及结构化处理器
"""

import os
import json
import re
import logging
import hashlib
from typing import TypedDict, List, Dict, Any, Set
from app.llm import chat_completion
from app.prompts.templates import CASE_DISTILL_PROMPT

from app.core.logger import get_logger
logger = get_logger(__name__)

class MetadataRegistry:
    """全局元数据注册表，管理库中发现的所有‘案件要素’与‘权责关键词’"""
    def __init__(self, storage_path: str = "data/metadata_registry.json"):
        self.storage_path = storage_path
        self.elements: Set[str] = set()
        self.keywords: Set[str] = set()
        self.load()

    def load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.elements = set(data.get("elements", []))
                    self.keywords = set(data.get("keywords", []))
            except Exception as e:
                logger.error(f"Failed to load metadata registry: {e}")

    def save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "elements": sorted(list(self.elements)),
                    "keywords": sorted(list(self.keywords))
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata registry: {e}")

    def register(self, elements: List[str], keywords: List[str]):
        new_e = [e for e in elements if e and e not in self.elements]
        new_k = [k for k in keywords if k and k not in self.keywords]
        if new_e or new_k:
            self.elements.update(new_e)
            self.keywords.update(new_k)
            self.save()

class ProcessedChunk(TypedDict):
    id: str
    content: str
    metadata: dict

class LawDataProcessor:
    """法律知识库处理器 - 采用 LLM 语义分拨架构"""

    def __init__(self, registry: MetadataRegistry = None):
        self.registry = registry or MetadataRegistry()

    @staticmethod
    def _truncate_footers(text: str) -> str:
        """物理截断：寻找文书终点锚点，防止内容噪音"""
        text = text.replace('\u3000', '  ')
        noise_markers = [
            r'\n\s*审\s*判\s*长\s+.*',
            r'\n\s*审\s*判\s*员\s+.*',
            r'\n\s*法\s*官\s*助\s*理\s+.*',
            r'\n\s*书\s*记\s*员\s+.*',
            r'\n\s*二\s*[〇○0][^行\n]*?年[^行\n]*?月[^行\n]*?日\s*$',
            r'\n\s*本案法律依据',
            r'©北大法宝'
        ]
        clean_end_pos = len(text)
        for marker in noise_markers:
            match = re.search(marker, text, re.DOTALL | re.MULTILINE)
            if match:
                if match.start() < clean_end_pos:
                    clean_end_pos = match.start()
        return text[:clean_end_pos].strip()

    @staticmethod
    def _clean_legal_text(text: str) -> str:
        """文书脱水，修复换行挤压"""
        text = re.sub(r'\n\s*-\s*\d+\s*-\s*\n', '\n', text)
        text = re.sub(r'([^\n。！？：;；])\n([^\n])', r'\1\2', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{2,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def _recursive_split(text: str, max_chars: int = 1200, overlap: int = 100) -> List[str]:
        """递归切片逻辑"""
        if len(text) <= max_chars:
            return [text]
        results = []
        start = 0
        while start < len(text):
            end = start + max_chars
            if end >= len(text):
                results.append(text[start:])
                break
            chunk = text[start:end]
            split_idx = -1
            for separator in ["\n\n", "\n", "。"]:
                found = chunk.rfind(separator)
                if found != -1 and found > max_chars // 2:
                    split_idx = found + len(separator)
                    break
            if split_idx == -1: split_idx = max_chars
            results.append(text[start:start+split_idx])
            start += (split_idx - overlap)
        return results

    def _detect_case_tier(self, case_name: str, text: str) -> str:
        """识别文书审级"""
        combined = (case_name + text[:1500]).lower()
        if any(w in combined for w in ["再审", "提审", "申请人称", "民申", "民再"]): return "TIER_RETRY"
        if any(w in combined for w in ["二审", "上诉", "终审", "民终"]): return "TIER_2ND"
        return "TIER_1ST"

    @staticmethod
    def _get_court_level(court_name: str) -> int:
        if "最高" in court_name: return 1
        if "高级" in court_name: return 2
        return 3 if "中级" in court_name else 4

    def _extract_fixed_metadata(self, text: str) -> Dict[str, Any]:
        """从文书头部正则提取固定格式的要素与关键词"""
        header_text = text[:2000]
        meta = {
            "cause_of_action": "", "case_id": "", "court_name": "",
            "keywords": [], "elements": []
        }
        if m := re.search(r'案由：\s*(.*?)(?=\s*\n|\s*案\s*号)', header_text):
            meta["cause_of_action"] = m.group(1).strip()
        if m := re.search(r'案\s*号：\s*(.*?)(?=\s*\n)', header_text):
            meta["case_id"] = m.group(1).strip()
        if m := re.search(r'审理法院：\s*(.*?)(?=\s*\n)', header_text):
            meta["court_name"] = m.group(1).strip()
        if m := re.search(r'权责关键词：\s*(.*?)(?=\s*\n|\s*案件要素|\s*相关企业)', header_text):
            meta["keywords"] = [k.strip() for k in m.group(1).replace('　', ' ').split() if k.strip()]
        if m := re.search(r'案件要素：\s*(.*?)(?=\s*\n|\s*相关企业|\s*相关人员)', header_text):
            meta["elements"] = [e.strip() for e in m.group(1).replace('　', ' ').split() if e.strip()]
        
        self.registry.register(meta["elements"], meta["keywords"])
        return meta

    @staticmethod
    def _extract_article_number(article_text: str) -> str:
        match = re.match(r"^\s*(第[一二三四五六七八九十百千零\d]+条)", article_text)
        return match.group(1) if match else ""

    @staticmethod
    def _extract_references(content_text: str) -> Dict[str, List[str]]:
        """
        Extract referenced laws and article-level references from interpretation text.
        This supports deferred linking when law and interpretation are uploaded out of order.
        """
        law_refs = sorted(set(re.findall(r"《([^》]{2,80})》", content_text)))
        article_refs = sorted(
            set(
                re.findall(
                    r"《[^》]{2,80}》第[一二三四五六七八九十百千零\d]+条",
                    content_text
                )
            )
        )
        return {
            "related_laws": law_refs,
            "related_articles": article_refs
        }

    async def _llm_distill_pipeline(self, cleaned_text: str) -> Dict[str, Any]:
        """调用 LLM 进行语义逻辑分拨与提炼 (采用 structured JSON Mode)"""
        try:
            # 过滤掉开头辅助信息，聚焦正文
            content_to_distill = cleaned_text.split("//", 1)[1].strip() if "//" in cleaned_text[:1000] else cleaned_text
            messages = [
                {"role": "system", "content": CASE_DISTILL_PROMPT},
                {"role": "user", "content": f"请提炼以下文书正文：\n\n{content_to_distill[:12000]}"}
            ]
            
            response = chat_completion(messages, temperature=0.1, json_mode=True)
            data = json.loads(response)
            
            return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.error(f"LLM distillation failed (JSON Mode error): {e}")
            return {}

    async def process_court_case(self, case_name: str, full_text: str) -> List[ProcessedChunk]:
        """核心 RAG 管线：语义分拨 + 0 重叠物理切片"""
        cleaned_text = self._truncate_footers(full_text)
        meta = self._extract_fixed_metadata(cleaned_text)
        
        case_id = meta["case_id"] or re.sub(r'\(.*?\)', '', case_name).strip()
        court_name = meta["court_name"] or "未知法院"
        year_match = re.search(r'\((\d{4})\)', case_id)
        year = int(year_match.group(1)) if year_match else 2025
        
        tier = self._detect_case_tier(case_name, cleaned_text)
        court_level = self._get_court_level(court_name)
        
        distilled_data = await self._llm_distill_pipeline(cleaned_text)
        chunks = []
        
        # 结构化解包
        document_summary = distilled_data.get("document_summary", {})
        segments = distilled_data.get("logic_segments", [])
        
        # 提取全局胜负判定与引证
        final_verdict = document_summary.get("final_verdict", "未明确")
        is_plaintiff_win = document_summary.get("is_plaintiff_win", False)
        cited_laws = document_summary.get("cited_laws", [])
        
        summary_info_full = f"案号: {case_id} | 法院: {court_name} | {meta['cause_of_action']}\n全局裁判结果: {final_verdict} (原告胜诉: {is_plaintiff_win})"
        summary_info_partial = f"案号: {case_id} | 法院: {court_name} | {meta['cause_of_action']}"
        
        label_map = {
            "attack": "[⚠️单方主张参考]", "counter": "[⚠️单方抗辩参考]",
            "fact": "[客观事实认定]", "reasoning": "[法理裁决说理]", "verdict": "[最终裁判指令]"
        }

        for sec_idx, section in enumerate(segments):
            logic = section.get("segment_type", "unknown").split(' ')[0]
            body = section.get("content", "")
            if not body: continue
            
            # 优化：主张、抗辩、裁判结果 均不注入全局结论，避免信息冗余或泄露结果给推理过程
            if logic in ["attack", "counter", "verdict"]:
                summary_info = summary_info_partial
            else:
                summary_info = summary_info_full
            
            header = f"{label_map.get(logic, f'[{logic}]')}\n背景: {summary_info}\n板块要点: {section.get('summary', '')}\n"
            sub_points = self._recursive_split(body, max_chars=3000, overlap=0)
            
            for p_idx, sub_text in enumerate(sub_points):
                # 仅将实质内容（含 Part ID）放入 content，不影响 Embedding 的语义精度
                content = sub_text + (f" (Part {p_idx+1})" if len(sub_points) > 1 else "")
                chunk_id = hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]
                
                chunks.append({
                    "id": f"case_{case_id}_{logic}_s{sec_idx}_p{p_idx}_{chunk_id}",
                    "content": content,
                    "metadata": {
                        "case_id": case_id, "year": year, "court_name": court_name,
                        "court_level": court_level, "section_type": logic,
                        "logic_type": logic, "case_name": case_name, "tier": tier,
                        "elements": meta["elements"] if meta["elements"] else ["无"], 
                        "keywords": meta["keywords"] if meta["keywords"] else ["无"],
                        "cause_of_action": meta["cause_of_action"],
                        "is_plaintiff_win": is_plaintiff_win,
                        "cited_laws": cited_laws if cited_laws else ["未提及"],
                        "summary": section.get("summary", ""),
                        "part_idx": p_idx,
                        "header_context": header # 存入元数据，仅在召回转换时还原
                    }
                })
        return chunks

    def process_law_article(self, law_name: str, content_text: str, doc_subtype: str = "law") -> List[ProcessedChunk]:
        """解析法条信息（保持原有逻辑）"""
        clean_law_name = re.sub(r'\.\w+$', '', law_name).strip()
        normalized_subtype = "interpretation" if doc_subtype == "interpretation" else "law"
        authority_level = 2 if normalized_subtype == "interpretation" else (
            1 if "法" in clean_law_name and "中华人民共和国" in clean_law_name else 3
        )
        refs = self._extract_references(content_text) if normalized_subtype == "interpretation" else {
            "related_laws": [],
            "related_articles": []
        }
        chunks = []
        matches = list(re.finditer(r'第[一二三四五六七八九十百千]+条[ 　、：:]?', content_text))
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i+1].start() if i + 1 < len(matches) else len(content_text)
            text = content_text[start:end].strip()
            if not text: continue
            article_num = self._extract_article_number(text)
            content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]
            chunks.append({
                "id": f"law_{clean_law_name}_{i}_{content_hash}",
                "content": f"[裁判依据: 《{clean_law_name}》]\n" + text,
                "metadata": {
                    "law_name": clean_law_name,
                    "document_name": clean_law_name,
                    "type": "law_article",
                    "doc_subtype": normalized_subtype,
                    "authority_level": authority_level,
                    "article_num": article_num,
                    "related_laws": refs["related_laws"],
                    "related_articles": refs["related_articles"]
                }
            })
        if not chunks and content_text.strip():
            content_hash = hashlib.sha256(content_text.encode('utf-8')).hexdigest()[:8]
            chunks.append({
                "id": f"law_{clean_law_name}_0_{content_hash}",
                "content": f"[裁判依据: 《{clean_law_name}》]\n" + content_text.strip(),
                "metadata": {
                    "law_name": clean_law_name,
                    "document_name": clean_law_name,
                    "type": "law_article",
                    "doc_subtype": normalized_subtype,
                    "authority_level": authority_level,
                    "article_num": "",
                    "related_laws": refs["related_laws"],
                    "related_articles": refs["related_articles"]
                }
            })
        return chunks
