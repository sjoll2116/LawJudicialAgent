"""
法律数据入库脚本 (支持 Word 与 增量入库)
扫描 resources/ 目录下的 .txt 和 .docx 文件，
利用 MD5 哈希确保文件不被重复存入 SeekDB。
"""

import os
import sys
import logging
import hashlib
import json
from pathlib import Path
from docx import Document
import re

from llama_index.core import SimpleDirectoryReader

# 确保能导入 app 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.law_processor import LawDataProcessor
from app.rag.retriever import HybridRetriever

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HISTORY_FILE = "data/ingest_history.json"

def get_file_hash(file_path: Path) -> str:
    """计算文件内容的 MD5 哈希"""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def read_text_from_file(file_path: Path) -> str:
    """利用 LlamaIndex 统一读取 .txt, .docx, .pdf 等格式，同时保留 .doc 兼容层"""
    suffix = file_path.suffix.lower()
    
    # 针对旧版 .doc 维持 win32com 兼容层 (LlamaIndex 在 Windows 上原生处理 .doc 较弱)
    if suffix == ".doc":
        import win32com.client
        word = None
        doc = None
        try:
            word = win32com.client.Dispatch("Word.Application")
            word.Visible = False
            abs_path = str(file_path.resolve())
            doc = word.Documents.Open(abs_path)
            content = doc.Content.Text
            return content.replace('\r', '\n')
        except Exception as e:
            logger.error(f"解析 .doc 失败: {str(e)}")
            return ""
        finally:
            if doc: doc.Close(False)
            if word: word.Quit()
    
    # 其余现代格式
    try:
        # 第一层：尝试 LlamaIndex (标准引擎)
        reader = SimpleDirectoryReader(input_files=[str(file_path)])
        documents = reader.load_data()
        content = "\n".join([doc.text for doc in documents]).strip()
        if content: return content
    except Exception as e:
        logger.warning(f"LlamaIndex 级联失效 {file_path.name}: {str(e)}")
    
    # 第二层：针对 .docx 的强行降级 (使用 docx2txt)
    if suffix == ".docx":
        try:
            import docx2txt
            content = docx2txt.process(str(file_path))
            if content.strip(): 
                logger.info(f"✅ 使用 docx2txt 强行找回内容: {file_path.name}")
                return content
        except Exception as e:
            logger.error(f"docx2txt 强行处理失败: {e}")

    # 第三层：暴力暴力读取 (针对 TXT 或损坏文件)
    try:
        for enc in ['utf-8', 'gbk', 'utf-16']:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except:
                continue
    except:
        pass
        
    return ""

def extract_metadata_from_case(content: str, file_stem: str) -> tuple[str, int, str]:
    """
    从文书正文中提取法院、年份、案号。
    法律文书标准结构：
    第一行：法院名称
    第三行：案号 (yyyy) 字号...号
    """
    lines = [L.strip() for L in content.split("\n") if L.strip()]
    
    court_name = "未知法院"
    year = 2025
    case_id = file_stem
    
    # 1. 提取法院 (通常在第一行，以 '法院' 结尾)
    for i in range(min(5, len(lines))):
        if "法院" in lines[i]:
            court_name = lines[i]
            break
            
    # 2. 提取案号和年份 (使用正则查找：(年份)案号)
    # 支持：(2023)最高法...、[2023]...、2023...号、以及全角括号 （2023）
    # 优先匹配带括号的案号，因为最标准
    case_no_ptn = re.compile(r'[（\(\[](\d{4})[）\)\]][^号\n]+?\d+号')
    
    # 尝试在前 10 行匹配
    found_match = None
    for i in range(min(15, len(lines))):
        match = case_no_ptn.search(lines[i])
        if match:
            found_match = match
            case_id = lines[i]
            break
            
    # 如果前 15 行没找到，全文本搜索一次
    if not found_match:
        full_match = case_no_ptn.search(content)
        if full_match:
            found_match = full_match
            # 提取包含该案号的一整行作为 case_id
            start = content.rfind("\n", 0, full_match.start()) + 1
            end = content.find("\n", full_match.end())
            case_id = content[start:end].strip() if end != -1 else content[start:].strip()

    if found_match:
        year = int(found_match.group(1))
            
    return court_name, year, case_id

def load_ingest_history() -> dict:
    """加载已入库文件的历史记录"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_ingest_history(history: dict):
    """保存入库历史"""
    os.makedirs("data", exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def run_ingestion():
    processor = LawDataProcessor()
    retriever = HybridRetriever()
    history = load_ingest_history()
    
    base_path = Path("resources")
    law_dir = base_path / "law_articles"
    case_dir = base_path / "court_cases"

    # 1. 自动处理法条
    if law_dir.exists():
        logger.info(f"正在扫描法条目录: {law_dir}")
        for file_path in law_dir.glob("*"):
            if file_path.suffix.lower() not in [".txt", ".docx", ".doc"]:
                continue
                
            file_hash = get_file_hash(file_path)
            if history.get(str(file_path)) == file_hash:
                logger.info(f"跳过已存在法典: {file_path.name}")
                continue

            law_name = file_path.stem
            logger.info(f">>> 增量处理法典: {law_name}")
            content = read_text_from_file(file_path)
            
            chunks = processor.process_law_article(
                law_name=law_name,
                content_text=content
            )
            if chunks:
                retriever.ingest_law_articles(
                    ids=[c["id"] for c in chunks],
                    documents=[c["content"] for c in chunks],
                    metadatas=[c["metadata"] for c in chunks]
                )
                history[str(file_path)] = file_hash
                logger.info(f"✅ 成功入库法典: {law_name}, 切块数: {len(chunks)}")
            else:
                logger.warning(f"⚠️ 法典 {law_name} 切割结果为空，请检查文件格式是否符合 [第X条] 规范。")

    # 2. 自动处理案例（实现镜像切片）
    if case_dir.exists():
        logger.info(f"正在扫描案例目录: {case_dir}")
        for file_path in case_dir.glob("*"):
            if file_path.suffix.lower() not in [".txt", ".docx", ".doc"]:
                continue
                
            file_hash = get_file_hash(file_path)
            if history.get(str(file_path)) == file_hash:
                logger.info(f"跳过已存在案例: {file_path.name}")
                continue

            file_name = file_path.stem
            logger.info(f">>> 增量处理案例文书: {file_name}")
            content = read_text_from_file(file_path)
            
            # 优先从正文提取元数据
            extracted_court, extracted_year, extracted_id = extract_metadata_from_case(content, file_name)
            
            # 元数据决策逻辑
            parts = file_name.split("_")
            if "号" in extracted_id and extracted_year != 2025:
                # 如果正文提取到了看起来很像案号的内容，强制使用正文结果
                court_name = extracted_court
                year = extracted_year
                case_id = extracted_id
                logger.info(f"优先选用正文元数据 -> 案号: {case_id}")
            elif len(parts) >= 3:
                # 否则如果文件名符合约定格式，使用文件名
                court_name = parts[0]
                year = int(parts[1]) if parts[1].isdigit() else extracted_year
                case_id = parts[2]
            else:
                # 最后降级使用各种提取结果或默认值
                court_name = extracted_court
                year = extracted_year
                case_id = extracted_id
                
            logger.info(f"最终解析 -> 法院: {court_name}, 年份: {year}, 案号: {case_id}")
            
            import asyncio
            chunks = asyncio.run(processor.process_court_case(
                case_name=file_name,
                full_text=content
            ))
            
            if chunks:
                retriever.ingest_court_cases(
                    ids=[c["id"] for c in chunks],
                    documents=[c["content"] for c in chunks],
                    metadatas=[c["metadata"] for c in chunks]
                )
                history[str(file_path)] = file_hash

    save_ingest_history(history)
    logger.info("=" * 30)
    logger.info("✅ 增量入库任务完成。")
    logger.info(f"法条库总数: {retriever._law_vect.count()}")
    logger.info(f"案例库总数: {retriever._case_vect.count()}")
    logger.info("=" * 30)

if __name__ == "__main__":
    os.makedirs("./data/seekdb", exist_ok=True)
    run_ingestion()
