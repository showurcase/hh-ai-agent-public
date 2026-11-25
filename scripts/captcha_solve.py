"""Прочитать капчу hh.ru с картинки тем же решателем, что использует бот.

Запуск в контейнере:
    docker cp captcha.png hh-ai-agent:/tmp/captcha.png
    docker exec hh-ai-agent python /app/scripts/captcha_solve.py /tmp/captcha.png
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_analyzer import VacancyAnalyzer  # noqa: E402
from config import load_settings  # noqa: E402
from database import Database  # noqa: E402
from llm.factory import create_llm_provider  # noqa: E402


async def main(image_path: Path) -> None:
    settings = load_settings()
    database = Database(settings.database_path)
    database.init()
    provider = create_llm_provider(settings, database)
    try:
        analyzer = VacancyAnalyzer(settings, provider)
        answer = await analyzer.solve_captcha(image_path)
        print(f"answer={answer!r}")
    finally:
        await provider.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: captcha_solve.py <image>")
    asyncio.run(main(Path(sys.argv[1])))
