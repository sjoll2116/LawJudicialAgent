import asyncio
import os

import pytest

from app.rag.law_processor import LawDataProcessor


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1",
    reason="Set RUN_INTEGRATION=1 to run real distillation pipeline test.",
)
def test_paragraph_splitting_smoke():
    processor = LawDataProcessor()
    test_text = "原告诉称：被告违约。被告辩称：不存在违约。本院认为：支持部分诉请。"
    res = asyncio.run(processor.process_court_case("测试案例", test_text))
    assert isinstance(res, list)
