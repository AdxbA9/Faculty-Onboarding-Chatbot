"""UOS Faculty Handbook chatbot – backend package.

Public API:

    from handbook_bot import build_knowledge_base, answer_question, QAResult
"""
from .knowledge_base import KnowledgeBase, build_knowledge_base
from .qa import QAResult, answer_question

__all__ = [
    "KnowledgeBase",
    "QAResult",
    "answer_question",
    "build_knowledge_base",
]

__version__ = "1.0.0"
