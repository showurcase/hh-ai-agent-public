"""Один цикл поиска и автоотклика с логом в консоль.

Запускается вручную для отладки:
    docker compose stop hh-agent
    docker compose run --rm -v $PWD/scripts/run_cycle.py:/app/run_cycle.py:ro \
        --entrypoint python hh-agent /app/run_cycle.py 3
"""

from __future__ import annotations

import asyncio
import logging
import sys

from ai_analyzer import VacancyAnalyzer
from approval import ApprovalGuard, ApprovalService
from browser_backend import create_browser_backend
from config import load_settings
from database import Database
from hh_client import HHClient
from llm.factory import create_llm_provider
from main import run_search_cycle
from tg_bot import AgentControl


class LogTelegram:
    """Печатает уведомления вместо отправки в Telegram."""

    async def notify(self, text: str) -> None:
        print(f"[NOTIFY] {text}", flush=True)

    async def notify_analysis_failed(self, title: str, url: str, error_type: str) -> None:
        print(f"[ANALYSIS_FAILED] {error_type} {title}", flush=True)

    async def notify_questionnaire_required(self, title: str, url: str) -> None:
        print(f"[QUESTIONNAIRE] {title} {url}", flush=True)

    async def send_preview(self, vacancy, include_actions: bool) -> None:
        print(f"[PREVIEW] {vacancy.title}", flush=True)

    async def request_captcha(self, screenshot, title: str, timeout_seconds: int):
        print(f"[CAPTCHA_MANUAL] {title}", flush=True)
        return None

    async def notify_screenshot(self, path, caption: str) -> None:
        print(f"[SCREENSHOT] {path}", flush=True)


async def main() -> None:
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = load_settings()
    database = Database(settings.database_path)
    database.init()
    backend = create_browser_backend(settings)
    provider = create_llm_provider(settings, database)
    context = await backend.start()
    try:
        hh_client = HHClient(context, settings, database, ApprovalGuard(settings, database))
        analyzer = VacancyAnalyzer(settings, provider)
        hh_client.analyzer = analyzer

        async def solve_captcha(screenshot, title, timeout_seconds):
            return await analyzer.solve_captcha(screenshot)

        hh_client.captcha_solver = solve_captcha
        service = ApprovalService(settings, database, hh_client)
        control = AgentControl()
        run = await run_search_cycle(
            settings,
            database,
            hh_client,
            analyzer,
            LogTelegram(),
            control,
            approval_service=service,
            auto_apply_batch_limit=batch,
        )
        print(
            f"[CYCLE] state={run.state} applied={run.auto_applications} "
            f"cards={run.telegram_cards} errors={run.error_count}",
            flush=True,
        )
    finally:
        await provider.close()
        await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
