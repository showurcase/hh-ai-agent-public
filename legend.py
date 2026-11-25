from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

logger = logging.getLogger(__name__)

_SECTIONS = (
    "general_info",
    "project_1_dla",
    "project_2_qr",
    "roles_and_processes",
    "tech_stack",
    "self_presentation",
)


@dataclass(frozen=True)
class Legend:
    text: str = ""

    @property
    def available(self) -> bool:
        return bool(self.text)


def render_legend(data: Mapping[str, object]) -> str:
    parts: list[str] = []
    for section in _SECTIONS:
        value = data.get(section)
        if not value:
            continue
        parts.append(
            f"### {section}\n{json.dumps(value, ensure_ascii=False, indent=1)}"
        )
    return "\n\n".join(parts)


def load_legend(path: Path | str | None) -> Legend:
    if path is None:
        return Legend()
    legend_path = Path(path)
    if not legend_path.exists():
        logger.warning("legend_file_missing path=%s", legend_path)
        return Legend()
    try:
        data = json.loads(legend_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("legend_file_unreadable path=%s error=%s", legend_path, exc)
        return Legend()
    if not isinstance(data, dict):
        logger.warning("legend_file_invalid_root path=%s", legend_path)
        return Legend()
    return Legend(render_legend(data))


def salary_from_profile(value: str) -> str:
    """Human-readable salary answer for free-text form fields."""
    value = value.strip()
    if not value:
        return ""
    digits = "".join(char for char in value if char.isdigit())
    if not digits:
        return value
    amount = int(digits)
    return f"от {amount:,} рублей на руки".replace(",", " ")


def salary_digits(value: str) -> str:
    digits = "".join(char for char in value if char.isdigit())
    return digits or "250000"
