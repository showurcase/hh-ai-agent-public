import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from config import load_settings
from database import Database, VacancyStatus
from queue_page import (
    QueuePageServer,
    issue_token,
    render_page,
    token_is_valid,
)
from tests.test_config import VALID_ENV, write_profile


NOW = datetime(2026, 7, 26, 9, 0, tzinfo=UTC)
SECRET = "queue-secret"


def settings(tmp_path: Path, **environment: str):
    loaded = load_settings(
        profile_path=write_profile(tmp_path),
        environ={**VALID_ENV, "QUEUE_PAGE_ENABLED": "true",
                 "QUEUE_PAGE_PUBLIC_URL": "http://127.0.0.1:8787",
                 "QUEUE_PAGE_SECRET": SECRET, **environment},
    )
    return replace(loaded, database_path=tmp_path / "agent.db")


def pending(database: Database, job_id: str = "job-1", title: str = "Python developer") -> None:
    assert database.discover(
        job_id=job_id,
        title=title,
        company="Example",
        url=f"https://hh.ru/vacancy/{job_id}",
        description_hash="hash",
        search_query="Python",
        discovered_at=NOW,
    )
    assert database.request_approval(
        job_id=job_id,
        cover_letter="Letter",
        llm_decision=True,
        llm_reason="Relevant",
        confidence=0.9,
        now=NOW,
    )


def test_token_expires_and_rejects_tampering() -> None:
    token = issue_token(SECRET, now=1000, ttl=60)

    assert token_is_valid(SECRET, token, now=1030)
    assert not token_is_valid(SECRET, token, now=1061)
    assert not token_is_valid("other-secret", token, now=1030)
    assert not token_is_valid(SECRET, token + "0", now=1030)
    assert not token_is_valid(SECRET, "garbage", now=1030)


def test_page_lists_pending_vacancies_with_escaped_links(tmp_path: Path) -> None:
    app_settings = settings(tmp_path)
    database = Database(app_settings.database_path)
    database.init()
    pending(database, title="Аналитик <script>")

    page = render_page(database.pending(), token="tok")

    assert "Аналитик &lt;script&gt;" in page
    assert 'href="https://hh.ru/vacancy/job-1"' in page
    assert "/q/tok/done/job-1?as=applied" in page
    assert "/q/tok/done/job-1?as=skipped" in page
    assert "/q/tok/clear" in page


def test_page_shows_attention_note(tmp_path: Path) -> None:
    app_settings = settings(tmp_path)
    database = Database(app_settings.database_path)
    database.init()
    assert database.discover(
        job_id="job-1",
        title="Системный аналитик",
        company="Example",
        url="https://hh.ru/vacancy/job-1",
        description_hash="hash",
        search_query="Аналитик",
        discovered_at=NOW,
    )
    assert database.request_approval(
        job_id="job-1",
        cover_letter="Letter",
        llm_decision=True,
        llm_reason="Relevant",
        confidence=0.9,
        fit_summary="Опыт: backend-сервисы\nВнимание: работа в офисе в Новосибирске",
        now=NOW,
    )

    page = render_page(database.pending(), token="tok")

    assert "работа в офисе в Новосибирске" in page
    assert 'class="attention"' in page
    assert "backend-сервисы" not in page


def test_page_has_no_attention_block_without_note(tmp_path: Path) -> None:
    app_settings = settings(tmp_path)
    database = Database(app_settings.database_path)
    database.init()
    pending(database)

    page = render_page(database.pending(), token="tok")

    assert 'class="attention"' not in page


def test_page_exposes_letter_for_one_click_copy(tmp_path: Path) -> None:
    app_settings = settings(tmp_path)
    database = Database(app_settings.database_path)
    database.init()
    pending(database)

    page = render_page(database.pending(), token="tok")

    assert "Скопировать письмо" in page
    assert 'data-target="letter-job-1"' in page
    assert '<span class="letter-source" id="letter-job-1" hidden>Letter</span>' in page


def test_page_hides_letter_block_when_letter_is_empty(tmp_path: Path) -> None:
    app_settings = settings(tmp_path)
    database = Database(app_settings.database_path)
    database.init()
    assert database.discover(
        job_id="job-2",
        title="Analyst",
        company="Example",
        url="https://hh.ru/vacancy/job-2",
        description_hash="hash",
        search_query="Analyst",
        discovered_at=NOW,
    )
    assert database.request_approval(
        job_id="job-2",
        cover_letter="",
        llm_decision=True,
        llm_reason="Relevant",
        confidence=0.9,
        now=NOW,
    )

    page = render_page(database.pending(), token="tok")

    assert 'class="copy"' not in page
    assert "letter-source" not in page


def test_page_is_empty_when_nothing_pending(tmp_path: Path) -> None:
    page = render_page([], token="tok")

    assert "Очередь пуста" in page
    assert "/clear" not in page


def run(coroutine):
    return asyncio.run(coroutine)


def test_queue_page_marks_applied_and_skipped(tmp_path: Path) -> None:
    app_settings = settings(tmp_path)
    database = Database(app_settings.database_path)
    database.init()
    pending(database, "job-1")
    pending(database, "job-2")
    server = QueuePageServer(app_settings, database)
    token = issue_token(app_settings.queue_page_secret)

    async def scenario() -> None:
        client = TestClient(TestServer(server.build_app()))
        await client.start_server()
        try:
            page = await client.get(f"/q/{token}")
            assert page.status == 200
            assert "job-1" in await page.text()

            applied = await client.get(
                f"/q/{token}/done/job-1?as=applied", allow_redirects=False
            )
            assert applied.status == 303
            skipped = await client.get(
                f"/q/{token}/done/job-2?as=skipped", allow_redirects=False
            )
            assert skipped.status == 303
        finally:
            await client.close()

    run(scenario())

    assert database.get("job-1").status is VacancyStatus.APPLIED
    assert database.get("job-2").status is VacancyStatus.SKIPPED


def test_queue_page_rejects_expired_token(tmp_path: Path) -> None:
    app_settings = settings(tmp_path)
    database = Database(app_settings.database_path)
    database.init()
    pending(database)
    server = QueuePageServer(app_settings, database)
    expired = issue_token(app_settings.queue_page_secret, ttl=-60)

    async def scenario() -> int:
        client = TestClient(TestServer(server.build_app()))
        await client.start_server()
        try:
            response = await client.get(f"/q/{expired}")
            return response.status
        finally:
            await client.close()

    assert run(scenario()) == 403
    assert database.get("job-1").status is VacancyStatus.PENDING_APPROVAL


def test_clear_empties_the_queue(tmp_path: Path) -> None:
    app_settings = settings(tmp_path)
    database = Database(app_settings.database_path)
    database.init()
    pending(database, "job-1")
    pending(database, "job-2")
    server = QueuePageServer(app_settings, database)
    token = issue_token(app_settings.queue_page_secret)

    async def scenario() -> int:
        client = TestClient(TestServer(server.build_app()))
        await client.start_server()
        try:
            response = await client.post(
                f"/q/{token}/clear", allow_redirects=False
            )
            return response.status
        finally:
            await client.close()

    assert run(scenario()) == 303
    assert database.pending() == []


def test_link_is_not_built_when_page_is_disabled(tmp_path: Path) -> None:
    loaded = load_settings(
        profile_path=write_profile(tmp_path), environ=VALID_ENV
    )
    app_settings = replace(loaded, database_path=tmp_path / "agent.db")
    database = Database(app_settings.database_path)
    database.init()

    server = QueuePageServer(app_settings, database)

    assert not server.enabled
    assert server.url() == ""


def test_enabled_page_requires_public_url_and_secret(tmp_path: Path) -> None:
    from config import ConfigError

    with pytest.raises(ConfigError):
        load_settings(
            profile_path=write_profile(tmp_path),
            environ={**VALID_ENV, "QUEUE_PAGE_ENABLED": "true"},
        )
