"""Прочитать вакансию тем же кодом и тем же браузером, что использует бот.

Запуск при остановленном боте:
    docker compose stop
    docker compose run --rm --entrypoint python \
        -v ~/hh-ai-agent/scripts:/app/scripts:ro \
        hh-agent /app/scripts/probe_read.py <job_id>
    docker compose start
"""

from __future__ import annotations

import asyncio
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_analyzer import VacancyAnalyzer  # noqa: E402
from approval import ApprovalGuard  # noqa: E402
from browser_backend import create_browser_backend  # noqa: E402
from config import load_settings  # noqa: E402
from database import Database  # noqa: E402
from hh_client import HHClient, VacancySummary  # noqa: E402
from llm.factory import create_llm_provider  # noqa: E402


async def main(job_id: str) -> None:
    settings = load_settings()
    database = Database(settings.database_path)
    database.init()
    row = database.get(job_id)
    if row is None:
        raise SystemExit(f"unknown job_id={job_id}")
    summary = VacancySummary(job_id, row.title, row.url, row.search_query or "")
    backend = create_browser_backend(settings)
    context = await backend.start()
    provider = create_llm_provider(settings, database)
    try:
        guard = ApprovalGuard(settings, database)
        client = HHClient(context, settings, database, guard)
        analyzer = VacancyAnalyzer(settings, provider)
        client.analyzer = analyzer

        async def solve_captcha(screenshot: Path, title: str, _timeout: int):
            answer = await analyzer.solve_captcha(screenshot)
            print(f"captcha answer={answer!r}", flush=True)
            return answer

        client.captcha_solver = solve_captcha
        try:
            details = await client.read_vacancy(summary, solve_captcha)
        except Exception:
            print("read_vacancy raised:", flush=True)
            traceback.print_exc()
            return
        print("state=", details.state, flush=True)
        page = await context.new_page()
        await page.goto(row.url, wait_until="domcontentloaded", timeout=30_000)
        try:
            print("final_url=", page.url, flush=True)
            print("title=", await page.title(), flush=True)
            text = (await page.locator("body").first.inner_text())[:600]
            print("body=", text.replace("\n", " | "), flush=True)
            await page.screenshot(path="/tmp/probe-read.png")
            print("screenshot=/tmp/probe-read.png", flush=True)
        except Exception as exc:
            print("page dump failed:", type(exc).__name__, exc, flush=True)
        print("error=", details.error, flush=True)
        print("company=", details.company, flush=True)
        print("description_len=", len(details.description), flush=True)
        print("description_head=", details.description[:160].replace("\n", " | "), flush=True)
        print("failure_screenshot=", client.last_failure_screenshot, flush=True)
        await page.close()
    finally:
        await provider.close()
        await backend.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: probe_read.py <job_id>")
    asyncio.run(main(sys.argv[1]))
