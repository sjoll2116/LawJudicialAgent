
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.law_processor import LawDataProcessor

def test_paragraph_splitting():
    processor = LawDataProcessor()
    
    # Simulate a 2nd instance case text
    test_text = """
广西壮族自治区高级人民法院
民事判决书
(2021)桂04民终914号

上诉权请求：
一、撤销一审判决。
二、改判赔偿。

事实和理由：
一、一审程序错误。
张锡彩应当列为第三人。
而且金额巨大。

二、一审认定事实错误。
海俊清恶意阻却合同。
此外父子关系成立。

被上诉人辩称：
一、张锡彩不应加入。
二审不需要他。

本院经审理查明：
二〇〇九年九月十七日签订了合同。
款项已支付。

本院认为：
上诉主张不成立。
一、程序并无不当。
二、认定事实清楚。

判决如下：
驳回上诉，维持原判。
二〇二一年十二月八日
"""
    
    res = processor.process_court_case("测试二审判决书", "2021-914", 2021, "广西高院", test_text)
    
    print(f"Total chunks generated: {len(res)}")
    for i, c in enumerate(res):
        print(f"\n--- Chunk {i+1} ({c['metadata']['logic_type']}) ---")
        print(c['content'])

if __name__ == "__main__":
    test_paragraph_splitting()
