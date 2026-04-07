import docx
import os

files = [
    r"e:\Law-judge\temp_uploads\DCM贸易公司、四川瑞巨物联网科技有限公司等股权转让纠纷二审民事判决书(FBMCLI.C.317442702).docx",
    r"e:\Law-judge\temp_uploads\广西梧州永杏药业有限公司、海俊清等股权转让纠纷民事申请再审审查民事裁定书(FBMCLI.C.403077247).docx",
    r"e:\Law-judge\temp_uploads\冯三利与龙门汇成投资有限公司请求变更公司登记纠纷一审民事判决书(FBMCLI.C.321868582).docx"
]

for f in files:
    try:
        doc = docx.Document(f)
        basename = os.path.basename(f).replace(".docx", ".txt")
        out_path = os.path.join(r"e:\Law-judge\temp_uploads", basename)
        with open(out_path, "w", encoding="utf-8") as out:
            out.write("\n".join([p.text for p in doc.paragraphs]))
        print(f"Extracted {f} to {out_path}")
    except Exception as e:
        print(f"Failed {f}: {e}")
