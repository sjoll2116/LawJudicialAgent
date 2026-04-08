from app.rag.law_processor import LawDataProcessor
import os

processor = LawDataProcessor()
# 使用较短的文件以加快检查速度，或者直接使用 DCM
file_path = r"e:\Law-judge\temp_uploads\DCM贸易公司、四川瑞巨物联网科技有限公司等股权转让纠纷二审民事判决书(FBMCLI.C.317442702).txt"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

chunks = processor.process_court_case("DCM", "(2020)川民终1443号", 2021, "四川高院", text)

with open("chunks_debug.txt", "w", encoding="utf-8") as f:
    for c in chunks:
        if "论点:" in c["content"]:
            f.write(f"--- CHUNK ID: {c['id']} ---\n")
            f.write(c["content"])
            f.write("\n\n")
