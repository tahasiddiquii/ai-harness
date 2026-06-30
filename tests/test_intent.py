from ai_harness.schemas import Complexity, Intent
from ai_harness.stages.intent import classify


def test_arithmetic_is_tool_use():
    c = classify("What is 14 * (9 + 3)?")
    assert c.intent is Intent.TOOL_USE
    assert c.needs_tools is True


def test_plain_question_is_qa_and_needs_retrieval():
    c = classify("What is retrieval augmented generation?")
    assert c.intent is Intent.QA
    assert c.needs_retrieval is True


def test_code_request_is_code():
    c = classify("Write a Python function to reverse a string")
    assert c.intent is Intent.CODE


def test_short_greeting_is_chitchat_and_simple():
    c = classify("hello there")
    assert c.intent is Intent.CHITCHAT
    assert c.complexity is Complexity.SIMPLE


def test_reasoning_query_is_complex():
    c = classify(
        "Explain and compare hybrid RAG versus graph RAG and analyse the trade-offs step by step"
    )
    assert c.complexity is Complexity.COMPLEX
