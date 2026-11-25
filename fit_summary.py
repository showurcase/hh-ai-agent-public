from __future__ import annotations

import re


FIT_CATEGORIES = frozenset(("Опыт", "Навыки", "Задачи", "Формат", "Локация", "Внимание"))
FIT_SUMMARY_FALLBACK = "Цель: вакансия соответствует моему направлению развития."
_MARKDOWN = re.compile(
    r"```|`|!?\[[^\]]*\]\([^)]+\)|"
    r"^\s{0,3}(?:#{1,6}|>|[-+*]|\d+[.)])\s+|"
    r"(?:\*{1,3}|_{1,3}|~{2})\S(?:.*?\S)?(?:\*{1,3}|_{1,3}|~{2})",
    re.MULTILINE,
)


def normalize_fit_summary(raw: object) -> str:
    if not isinstance(raw, list):
        return FIT_SUMMARY_FALLBACK
    lines: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        category, text = item.get("category"), item.get("text")
        if (
            not isinstance(category, str)
            or category not in FIT_CATEGORIES
            or category in seen
            or not isinstance(text, str)
            or _MARKDOWN.search(text)
        ):
            continue
        text = " ".join(text.split())
        max_len = 250 if category == "Внимание" else 140
        if not text or len(text) > max_len:
            continue
        lines.append(f"{category}: {text}")
        seen.add(category)
        if len(lines) == 5:
            break
    if not lines:
        return FIT_SUMMARY_FALLBACK
    return "\n".join(lines)
