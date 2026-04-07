import os
import pytest

from app.rag.retriever import HybridRetriever


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1",
    reason="Set RUN_INTEGRATION=1 to run real RAG integration test.",
)
def test_rag_search_smoke():
    retriever = HybridRetriever()
    query = "外商投资 股权转让 协议效力"
    results = retriever.search_all(query_text=query, n_results=3)

    assert isinstance(results, dict)
    assert "law_articles" in results
    assert "court_cases" in results
