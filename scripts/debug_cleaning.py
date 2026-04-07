
import re

def debug_clean_legal_text(text: str):
    print(f"--- 原始字数: {len(text)} ---")
    
    # 1. 物理截断锚点 (Terminal Anchors)
    noise_markers = [
        r'\n\s*审\s*判\s*长\s+.*',
        r'\n\s*二\s*[〇○0].*?年\s*.*?月\s*.*?日.*',
        r'\n\s*本案法律依据',
        r'\n\s*同案由重要案例',
        r'©北大法宝',
        r'扫描二维码阅读原文'
    ]
    
    clean_end_pos = len(text)
    matched_marker = "NONE"
    
    for marker in noise_markers:
        match = re.search(marker, text, re.DOTALL | re.MULTILINE)
        if match:
            if match.start() < clean_end_pos:
                clean_end_pos = match.start()
                matched_marker = marker

    main_text = text[:clean_end_pos].strip()
    print(f"--- 触发锚点: '{matched_marker}' ---")
    print(f"--- 清洗后字数: {len(main_text)} ---")
    
    print("\n--- 清洗后末尾内容 (最后 150 字) ---")
    print(main_text[-150:])
    
    return main_text

# 模拟 Part 3 结尾片段
sample_p3 = """
综上所述，丰某公司的上诉理由不成立... 依照... 判决如下：
驳回上诉，维持原判。
二审案件受理费719079.55元由丰某公司(H某某某某某某d)负担。
 
 审 判 长 刘娟娟 审 判 员 王 乐 审 判 员 戈光应 二〇二五年三月二十四日 法官助理 周相杜 书 记 员 熊其涛
 
本案法律依据 
中华人民共和国涉外民事关系法律适用法 第14条1款 
更多
同案由重要案例 
江苏某建设有限公司...
"""

debug_clean_legal_text(sample_p3)
