
import sys
import os
import threading
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.retriever import HybridRetriever

def simulate_ingest(retriever, thread_id):
    print(f"Thread-{thread_id} starting ingest...")
    ids = [f"test_{thread_id}_{i}" for i in range(3)]
    docs = [f"content from thread {thread_id} part {i}" for i in range(3)]
    metas = [{"case_name": f"test_thread_{thread_id}", "case_id": f"id_{thread_id}"} for _ in range(3)]
    
    try:
        retriever.ingest_court_cases(ids, docs, metas)
        print(f"Thread-{thread_id} success!")
    except Exception as e:
        print(f"Thread-{thread_id} failed: {e}")

def main():
    print("=== 开始并发入库压力测试 ===")
    retriever = HybridRetriever()
    
    threads = []
    for i in range(5):
        t = threading.Thread(target=simulate_ingest, args=(retriever, i))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
    
    print("\n=== 测试完成 ===")
    print("检查结果：如果控制台没有出现 LockError，则说明线程锁生效了。")

if __name__ == "__main__":
    main()
