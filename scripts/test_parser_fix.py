import re
from app.rag.law_processor import LawDataProcessor

def test_law_parser_fixes():
    processor = LawDataProcessor()
    
    # 模拟包含“零”的条文和带有章节标题的杂质
    test_text = """
第八十四条　在证据可能灭失或者以后难以取得的情况下，当事人可以在诉讼过程中向人民法院申请保全证据，人民法院也可以主动采取保全措施。
第七章　期间、送达
第一节　期间

第一百零九条　人民法院对必须到庭的被告，经两次传票传唤，无正当理由拒不到庭的，可以拘传。
第八章　调解
"""
    
    print("Testing parser fixes...")
    chunks = processor.process_law_article("民事诉讼法", test_text)
    
    print(f"Total chunks generated: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        content = chunk["content"]
        article_num = chunk["metadata"]["article_num"]
        print(f"\n--- Chunk {i+1} (Article: {article_num}) ---")
        print(content)
        
        # 验证 1：不应包含章节标题
        assert "第七章" not in content
        assert "第一节" not in content
        assert "第八章" not in content
        print("✓ No noise headings found.")
        
    # 验证 2：应当成功切分出两个条文（第八十四条 和 第一百零九条）
    assert len(chunks) == 2
    assert chunks[0]["metadata"]["article_num"] == "第八十四条"
    assert chunks[1]["metadata"]["article_num"] == "第一百零九条"
    print("\n✓ Triple-digit article with '零' successfully split!")
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_law_parser_fixes()
