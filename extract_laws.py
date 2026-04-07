import os
import pypdf
import docx

laws_dir = r"e:\Law-judge\法律条款"
output_file = r"e:\Law-judge\extracted_laws.txt"

def extract():
    with open(output_file, "w", encoding="utf-8") as out:
        # PDF
        pdf_path = os.path.join(laws_dir, "外商投资法.pdf")
        if os.path.exists(pdf_path):
            try:
                reader = pypdf.PdfReader(pdf_path)
                out.write("==== 《中华人民共和国外商投资法》 ====\n")
                for page in reader.pages:
                    text = page.extract_text()
                    out.write(text + "\n")
            except Exception as e:
                out.write(f"Error PDF: {e}\n")

        # DOCX
        docx_path = os.path.join(laws_dir, "外商投资法司法解释.docx")
        if os.path.exists(docx_path):
            try:
                doc = docx.Document(docx_path)
                out.write("\n==== 《最高人民法院关于适用〈中华人民共和国外商投资法〉若干问题的解释》 ====\n")
                for para in doc.paragraphs:
                    out.write(para.text + "\n")
            except Exception as e:
                out.write(f"Error DOCX: {e}\n")

extract()
