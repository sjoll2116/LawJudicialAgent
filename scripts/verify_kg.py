import sqlite3
import os

db_path = "e:/Law-judge/data/knowledge_graph.sqlite"
if not os.path.exists(db_path):
    print("DB file not found.")
else:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables: {tables}")
    
    # Count rows
    for table in tables:
        t_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {t_name}")
        count = cursor.fetchone()[0]
        print(f"Table {t_name}: {count} rows")
    
    conn.close()
