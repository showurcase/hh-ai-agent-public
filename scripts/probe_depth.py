"""Сколько новых вакансий лежит глубже первой страницы выдачи.

Запуск при остановленном боте:
    docker compose stop
    docker compose run --rm --entrypoint python \
        -v ~/hh-ai-agent/scripts:/app/scripts:ro \
        hh-agent /app/scripts/probe_depth.py "системный аналитик" 12
    docker compose start
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from browser_backend import create_browser_backend  # noqa: E402
from config import load_settings  # noqa: E402
from database import Database  # noqa: E402
from hh_client import _vacancy_id  # noqa: E402


async def main(query: str, pages: int) -> None:
    settings = load_settings()
    database = Database(settings.database_path)
    database.init()
    backend = create_browser_backend(settings)
    context = await backend.start()
    page = await context.new_page()
    known = 0
    fresh = 0
    seen: set[str] = set()
    try:
        for number in range(pages):
            params = {"text": query, "page": number}
            await page.goto(
                f"https://hh.ru/search/vacancy?{urlencode(params)}",
                wait_until="domcontentloaded",
                timeout=30_000,
            )
            await asyncio.sleep(2)
            cards = await page.locator('a[data-qa="serp-item__title"]').all()
            page_known = 0
            page_fresh = 0
            for card in cards:
                href = await card.get_attribute("href")
                job_id = _vacancy_id(href or "")
                if not job_id or job_id in seen:
                    continue
                seen.add(job_id)
                if database.get(job_id) is not None:
                    known += 1
                    page_known += 1
                else:
                    fresh += 1
                    page_fresh += 1
            print(
                f"стр.{number}: карточек={len(cards)} новых={page_fresh} известных={page_known}",
                flush=True,
            )
            if not cards:
                break
        print(f"итого: уникальных={len(seen)} новых={fresh} известных={known}", flush=True)
    finally:
        await page.close()
        await backend.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: probe_depth.py <query> [pages]")
    pages = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    asyncio.run(main(sys.argv[1], pages))
