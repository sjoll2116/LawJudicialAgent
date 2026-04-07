"""
SeekDB (ChromaDB + Whoosh) 数据删除工具
用法：
  py scripts/delete_data.py --case "案号"    - 删除特定案号的所有切块
  py scripts/delete_data.py --law "法典名"  - 删除特定法典的所有切块
  py scripts/delete_data.py --path "路径"   - 仅删除入库历史记录 (以便重新入库)
"""

import os
import sys
import json
import argparse
from pathlib import Path

# 确保能导入 app 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.retriever import HybridRetriever

HISTORY_FILE = "data/ingest_history.json"

def delete_by_case(retriever: HybridRetriever, case_id: str):
    """按案号删除案例库数据，并同步清理历史记录"""
    print(f"\n[1/3] 正在从 ChromaDB 查询案号: {case_id} ...")
    results = retriever._case_vect.get(where={"case_id": case_id})
    ids = results["ids"]
    metatas = results["metadatas"]
    
    related_case_names = set()
    if metatas:
        related_case_names = {m.get("case_name") for m in metatas if m.get("case_name")}
    
    if not ids:
        print(f"⚠️ 未在 ChromaDB 中找到案号 '{case_id}'。")
    else:
        retriever._case_vect.delete(ids=ids)
        print(f"✅ 已成功从向量库移除 {len(ids)} 个数据片段。")

    print(f"[2/3] 正在从 Whoosh 索引中移除...")
    writer = retriever._case_fts.writer()
    for doc_id in ids:
        writer.delete_by_term('id', doc_id)
    writer.commit()
    print(f"✅ Whoosh 全文索引已清理。")

    print(f"[3/3] 正在同步清理入库历史记录...")
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
        
        new_history = {}
        removed_paths = []
        for path, file_hash in history.items():
            # 匹配逻辑：如果路径包含 case_id 或者包含搜索到的 case_name
            should_remove = (case_id in path)
            for cn in related_case_names:
                if cn and cn in path:
                    should_remove = True
                    break
            
            if should_remove:
                removed_paths.append(path)
                continue
            new_history[path] = file_hash
        
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(new_history, f, ensure_ascii=False, indent=2)
        
        if removed_paths:
            for p in removed_paths:
                print(f"✅ 已同步移除历史记录: {p}")
        else:
            print("ℹ️ 未在历史记录中找到关联路径（可能已被手动清理过）。")
    
    print(f"\n✨ 案号 '{case_id}' 已彻底清理，现在你可以重新入库该文件。")

def delete_by_law(retriever: HybridRetriever, law_name: str):
    """按法典名删除法条库数据"""
    print(f"\n[1/3] 正在从 ChromaDB 删除法典: {law_name} ...")
    results = retriever._law_vect.get(where={"law_name": law_name})
    ids = results["ids"]
    
    if not ids:
        print(f"⚠️ 未在 ChromaDB 中找到法典 '{law_name}'。")
    else:
        retriever._law_vect.delete(ids=ids)
        print(f"✅ 已删除 ChromaDB 中的 {len(ids)} 个法条切块。")

    print(f"[2/3] 正在从 Whoosh 索引中移除...")
    writer = retriever._law_fts.writer()
    for doc_id in ids:
        writer.delete_by_term('id', doc_id)
    writer.commit()
    print(f"✅ Whoosh 索引已清理。")

    print(f"[3/3] 正在清理记录...")
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
        new_history = {k: v for k, v in history.items() if law_name not in k}
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(new_history, f, ensure_ascii=False, indent=2)
        print("✅ 历史记录已更新。")

def clear_path_history(path_str: str):
    """仅从历史记录中删除路径，以便重新入库"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
        
        # 尝试精确匹配或部分匹配
        keys_to_del = [k for k in history.keys() if path_str in k]
        if not keys_to_del:
            print(f"❌ 历史记录中未找到包含 '{path_str}' 的路径。")
            return
            
        for k in keys_to_del:
            del history[k]
            print(f"✅ 已移除历史记录: {k}")
            
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print("🚀 现在你可以重新运行 ingest_legal_data.py 来处理该文件了。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SeekDB 数据删除工具")
    parser.add_argument("--case", type=str, help="要删除的案号")
    parser.add_argument("--law", type=str, help="要删除的法典名")
    parser.add_argument("--path", type=str, help="要从历史记录中移除的文件路径片段")
    
    args = parser.parse_args()
    
    if not any([args.case, args.law, args.path]):
        parser.print_help()
        sys.exit(0)
        
    if args.path and not (args.case or args.law):
        clear_path_history(args.path)
    else:
        retriever = HybridRetriever()
        if args.case:
            delete_by_case(retriever, args.case)
        if args.law:
            delete_by_law(retriever, args.law)
