import os
import threading
import pytest

from app.rag.retriever import HybridRetriever


pytestmark = pytest.mark.integration


def _simulate_ingest(retriever, thread_id, errors):
    ids = [f"test_{thread_id}_{i}" for i in range(3)]
    docs = [f"content from thread {thread_id} part {i}" for i in range(3)]
    metas = [{"case_name": f"test_thread_{thread_id}", "case_id": f"id_{thread_id}"} for _ in range(3)]
    try:
        retriever.ingest_court_cases(ids, docs, metas)
    except Exception as e:
        errors.append(str(e))


@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1",
    reason="Set RUN_INTEGRATION=1 to run DB write concurrency test.",
)
def test_concurrent_ingest_smoke():
    retriever = HybridRetriever()
    errors = []

    threads = [threading.Thread(target=_simulate_ingest, args=(retriever, i, errors)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrency errors: {errors}"
