"""Detection of vacancies that touch AI, LLM or GenAI work."""

from __future__ import annotations

import re

AI_PATTERNS: tuple[str, ...] = (
    # LLM и языковые модели
    r"\bllms?\b",
    r"large\s+language\s+model",
    r"языков\w+\s+модел",
    r"\bgenai\b",
    r"генеративн\w+",
    r"\brag\b",
    r"\bgpt\b",
    r"chat\s*gpt",
    r"\bopenai\b",
    r"yandexgpt",
    r"\bgigachat\b",
    r"\bdeepseek\b",
    r"\bclaude\b",
    r"\bllama\b",
    r"\bqwen\b",
    r"\bmistral\b",
    r"\bgemini\b",
    # агенты, промпты, протоколы
    r"\bai-?агент",
    r"\bai-?agent",
    r"\bагентн",
    r"\bagents?\b",
    r"prompt\s*engineering",
    r"\bпромпт",
    r"\bmcp\b",
    # общие понятия
    r"искусственн\w+\s+интеллект",
    r"\bии\b",
    r"\bai\b",
    r"\bнейросет",
    r"\bнейронн",
    r"\bmlops\b",
    r"машинн\w+\s+обучен",
    r"machine\s+learning",
    r"\bml\b",
    r"deep\s+learning",
    r"глубок\w+\s+обучен",
    # прикладные направления
    r"\bnlp\b",
    r"computer\s*vision",
    r"компьютерн\w+\s+зрен",
    r"эмбеддинг",
    r"embedding",
    r"fine-?tun",
    r"дообучен\w+\s+модел",
    r"трансформер",
    r"vector\s*db",
    r"векторн\w+\s+баз",
    r"langchain",
    r"llamaindex",
    r"pgvector",
)

_AI_RE = re.compile("|".join(f"(?:{pattern})" for pattern in AI_PATTERNS), re.IGNORECASE)


def is_ai_llm_vacancy(*parts: str) -> bool:
    """True when the text mentions AI/LLM work in any form."""
    text = " ".join(part for part in parts if part)
    return bool(_AI_RE.search(text))
