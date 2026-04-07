import os

output_file = r"e:\Law-judge\extracted_laws.txt"
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        content = f.read()
        print(f"File size: {len(content)} characters")
        print("\n--- CONTENT PREVIEW ---\n")
        print(content[:2000])
else:
    print("File not found!")
