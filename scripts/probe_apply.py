"""Прогнать настоящий путь автоотклика по конкретной вакансии.

Возвращает вакансию в очередь и вызывает ApprovalService.auto_apply —
тот же код, что работает в боте, со всеми логами.

    docker compose run --rm -v $PWD/scripts/probe_apply.py:/app/probe_apply.py:ro \
        --entrypoint python hh-agent /app/probe_apply.py 139232012
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import sys

from ai_analyzer import VacancyAnalyzer
from approval import ApprovalGuard, ApprovalService
from browser_backend import create_browser_backend
from config import load_settings
from database import Database
from hh_client import HHClient
from llm.factory import create_llm_provider


async def main(job_id: str) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    settings = load_settings()
    database = Database(settings.database_path)
    database.init()
    with sqlite3.connect(settings.database_path) as connection:
        connection.execute(
            """
            UPDATE vacancies SET status = 'pending_approval',
                approved_at = NULL, applying_at = NULL, permit_hash = NULL,
                submit_attempted_at = NULL, applied_at = NULL, error_text = ''
            WHERE id = ?
            """,
            (job_id,),
        )
    backend = create_browser_backend(settings)
    provider = create_llm_provider(settings, database)
    context = await backend.start()
    try:
        hh_client = HHClient(context, settings, database, ApprovalGuard(settings, database))
        analyzer = VacancyAnalyzer(settings, provider)
        hh_client.analyzer = analyzer

        async def solve_captcha(screenshot, title, timeout_seconds):
            answer = await analyzer.solve_captcha(screenshot)
            print(f"[captcha answer] {answer!r}", flush=True)
            return answer

        hh_client.captcha_solver = solve_captcha
        service = ApprovalService(settings, database, hh_client)
        result = await service.auto_apply(job_id)
        print(f"[result] ok={result.ok} message={result.message}", flush=True)
        vacancy = database.get(job_id)
        print(f"[state] {vacancy.status.value} error={vacancy.error_text!r}", flush=True)
    finally:
        await provider.close()
        await backend.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
