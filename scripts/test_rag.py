from app.rag.retriever import HybridRetriever
import json

def test():
    retriever = HybridRetriever()
    query = "马来西亚 股权转让 退股协议 是否有效"
    
    print(f"--- 正在执行 RRF 混合检索测试: {query} ---")
    results = retriever.search_all(query_text=query, n_results=3)
    
    # 验证法条召回
    print(f"\n[法条召回数量]: {len(results['law_articles'])}")
    for r in results['law_articles']:
        print(f"  - {r.metadata.get('law_name')}: Score {r.score:.4f}")

    # 验证判例召回与序列补全
    print(f"\n[判例召回数量]: {len(results['court_cases'])}")
    for r in results['court_cases']:
        print(f"  - [{r.metadata.get('case_name')}]: Score {r.score:.4f}")
        print(f"    内容预览(前100字): {r.content[:100]}...")
        
    # 验证 Prompt 格式化
    context = retriever.format_context_for_prompt(results)
    print("\n\n--- 最终喂给 AI 的 Context 预览 ---")
    print(context[:500])

if __name__ == "__main__":
    test()
