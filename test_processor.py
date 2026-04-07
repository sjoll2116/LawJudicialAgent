from app.rag.law_processor import LawDataProcessor
import os

def test_dcm_2nd():
    processor = LawDataProcessor()
    file_path = r"e:\Law-judge\temp_uploads\DCM贸易公司、四川瑞巨物联网科技有限公司等股权转让纠纷二审民事判决书(FBMCLI.C.317442702).txt"
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    case_name = "DCM贸易公司、四川瑞巨物联网科技有限公司等股权转让纠纷二审民事判决书"
    case_id = "(2020)川民终1443号"
    year = 2021
    court_name = "四川省高级人民法院"
    
    chunks = processor.process_court_case(case_name, case_id, year, court_name, text)
    
    print(f"Total chunks: {len(chunks)}")
    for chunk in chunks:
        if "attack" in chunk["metadata"]["logic_type"] and "论点:" in chunk["content"]:
            print(f"--- Attack Point Chunk ---")
            print(f"ID: {chunk['id']}")
            print(f"Content:\n{chunk['content']}")
            print(f"Metadata: {chunk['metadata']}")
            print("\n")
            break

if __name__ == "__main__":
    test_dcm_2nd()
