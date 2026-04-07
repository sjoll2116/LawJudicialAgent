"""
SeekDB (ChromaDB) 数据查看工具
用法：
  py scripts/view_chunks.py             - 列出所有已入库的法典和案例
  py scripts/view_chunks.py --case "案号" - 查看特定案例的详细切块
  py scripts/view_chunks.py --law "法典"  - 查看特定法典的详细切块
"""

import os
import sys
import io
import argparse
from typing import List, Dict, Any

# 强制 stdout 使用 UTF-8 编码，防止 Windows 终端报错
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 确保能导入 app 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.retriever import HybridRetriever

def list_collections(retriever: HybridRetriever):
    """显示数据库统计和概览"""
    print("\n" + "="*50)
    print("📊 SeekDB 数据库概览")
    print("="*50)
    
    # 1. 法条库统计
    law_data = retriever._law_vect.get()
    law_names = sorted(list(set([m.get("law_name") for m in law_data["metadatas"] if m.get("law_name")])))
    print(f"\n【法条库】 总切块数: {len(law_data['ids'])}")
    if law_names:
        print("已存入法典:")
        for name in law_names:
            count = sum(1 for m in law_data["metadatas"] if m.get("law_name") == name)
            print(f"  - {name} ({count} 个切块)")
    else:
        print("  (暂无法典数据)")
            
    # 2. 案例库统计
    case_data = retriever._case_vect.get()
    case_ids = sorted(list(set([m.get("case_id") for m in case_data["metadatas"] if m.get("case_id")])))
    print(f"\n【案例库】 总切块数: {len(case_data['ids'])}")
    if case_ids:
        print("已存入案号:")
        for cid in case_ids:
            count = sum(1 for m in case_data["metadatas"] if m.get("case_id") == cid)
            print(f"  - {cid} ({count} 个切块)")
    else:
        print("  (暂无案例数据)")
    print("\n" + "="*50)

def show_case_detail(retriever: HybridRetriever, case_id: str):
    """详细展示某个案例的切块内容"""
    print(f"\n>>> 正在查询案号: {case_id}")
    results = retriever._case_vect.get(where={"case_id": case_id})
    
    if not results["ids"]:
        print(f"❌ 未找到案号为 '{case_id}' 的数据")
        return
        
    print(f"✨ 共找到 {len(results['ids'])} 个切块内容：\n")
    
    # 排序：按照 logic_type 排序，让逻辑更顺畅
    # attack -> counter -> confirmed_facts -> law_rule_ref
    sort_order = {"attack": 0, "counter": 1, "confirmed_facts": 2, "law_rule_ref": 3}
    
    combined = []
    for i in range(len(results["ids"])):
        combined.append({
            "id": results["ids"][i],
            "content": results["documents"][i],
            "meta": results["metadatas"][i]
        })
    
    combined.sort(key=lambda x: sort_order.get(x["meta"].get("logic_type", ""), 99))
    
    for item in combined:
        logic_label = {
            "attack": "【原告主张】",
            "counter": "【被告抗辩】",
            "confirmed_facts": "【查明事实】",
            "law_rule_ref": "【本院认为】"
        }.get(item["meta"].get("logic_type"), "【其他部分】")
        
        print("-" * 30)
        print(f"{logic_label} ID: {item['id']}")
        print(f"内容摘要 ({len(item['content'])} 字):")
        print(item["content"])
        print("-" * 30 + "\n")

def show_law_detail(retriever: HybridRetriever, law_name: str):
    """详细展示某个法典的切块内容"""
    print(f"\n>>> 正在查询法典: {law_name}")
    results = retriever._law_vect.get(where={"law_name": law_name})
    
    if not results["ids"]:
        print(f"❌ 未找到名为 '{law_name}' 的数据")
        return
        
    print(f"✨ 共找到 {len(results['ids'])} 条法条：\n")
    for i in range(len(results["ids"])):
        print(f"[{i+1}] {results['ids'][i]}")
        print(results["documents"][i])
        print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SeekDB 数据查看工具")
    parser.add_argument("--case", type=str, help="指定要查看的案号")
    parser.add_argument("--law", type=str, help="指定要查看的法典名")
    
    args = parser.parse_args()
    
    retriever = HybridRetriever()
    
    if args.case:
        show_case_detail(retriever, args.case)
    elif args.law:
        show_law_detail(retriever, args.law)
    else:
        list_collections(retriever)
        print("\n提示：使用 --case \"案号\" 或 --law \"法典名\" 查看详细切块内容。")
