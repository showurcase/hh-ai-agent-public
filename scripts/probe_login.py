"""Проверить, залогинен ли бот на hh.ru, и что видно на странице резюме.

Запуск при остановленном боте:
    docker compose stop
    docker compose run --rm --entrypoint python \
        -v ~/hh-ai-agent/scripts:/app/scripts:ro \
        hh-agent /app/scripts/probe_login.py
    docker compose start
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from approval import ApprovalGuard  # noqa: E402
from browser_backend import create_browser_backend  # noqa: E402
from config import load_settings  # noqa: E402
from database import Database  # noqa: E402
from hh_client import HHClient  # noqa: E402


async def main() -> None:
    settings = load_settings()
    database = Database(settings.database_path)
    database.init()
    backend = create_browser_backend(settings)
    context = await backend.start()
    page = await context.new_page()
    try:
        await page.goto(
            "https://hh.ru/applicant/resumes",
            wait_until="domcontentloaded",
            timeout=60_000,
        )
        await asyncio.sleep(3)
        print("url:", str(page.url)[:100], flush=True)
        print("title:", await page.title(), flush=True)
        body = (await page.locator("body").first.inner_text())[:500]
        print("body:", body.replace("\n", " | "), flush=True)
        markers = page.locator(
            '[data-qa="mainmenu_applicantProfile"], [data-qa="mainmenu_profile"], '
            '[data-qa="mainmenu_resumes"], [data-qa*="resume-title"]'
        )
        print("маркеров входа видно:", await markers.count(), flush=True)
        client = HHClient(context, settings, database, ApprovalGuard(settings, database))
        print("ensure_login:", await client.ensure_login(), flush=True)
    finally:
        await page.close()
        await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
