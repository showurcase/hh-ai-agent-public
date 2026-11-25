import asyncio
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.methods import SendMessage

from config import load_settings
from database import Database
from tests.test_config import VALID_ENV, write_profile
from tg_bot import AgentControl, TelegramService


NOW = datetime(2026, 7, 26, 9, 0, tzinfo=UTC)


class FakeBot:
    def __init__(self, error: Exception | None = None):
        self.messages: list[dict] = []
        self.edits: list[dict] = []
        self.error = error
        self.edit_error: Exception | None = None

    async def send_message(self, **kwargs):
        if self.error:
            raise self.error
        self.messages.append(kwargs)
        return SimpleNamespace(message_id=len(self.messages))

    async def edit_message_text(self, **kwargs):
        self.edits.append(kwargs)
        if self.edit_error:
            raise self.edit_error
        return SimpleNamespace(message_id=kwargs["message_id"])


class FakeApprovalService:
    async def approve_and_apply(self, job_id: str, user_id: int):
        raise AssertionError("approval must not be called by command tests")

    def skip(self, job_id: str, user_id: int):
        raise AssertionError("skip must not be called by command tests")


def service(
    tmp_path: Path,
    app_mode: str = "dry_run",
    error: Exception | None = None,
):
    settings = load_settings(
        profile_path=write_profile(tmp_path),
        environ={**VALID_ENV, "TG_USER_ID": "42", "APP_MODE": app_mode},
    )
    settings = replace(settings, database_path=tmp_path / "agent.db")
    database = Database(settings.database_path)
    database.init()
    control = AgentControl()
    bot = FakeBot(error)
    telegram = TelegramService(
        settings,
        database,
        FakeApprovalService(),
        control,
        bot=bot,
        now_factory=lambda: NOW,
    )
    return telegram, database, control, bot


def add_preview_vacancy(database: Database) -> None:
    assert database.discover(
        job_id="job-1",
        title="Python <Developer>",
        company="Example & Co",
        company_url="https://hh.ru/employer/123",
        url="https://example.com/vacancy/job-1",
        description_hash="hash",
        search_query="Python",
        discovered_at=NOW,
    )
    assert database.store_company_details("job-1", rating=4.7, reviews_count=128)
    assert database.request_approval(
        job_id="job-1",
        cover_letter="Hello <team>",
        llm_decision=True,
        llm_reason="Relevant & local",
        fit_summary="Опыт: backend-сервисы.\nНавыки: Python.",
        confidence=0.87,
        now=NOW,
    )


def test_polling_leaves_shutdown_signals_to_application(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    telegram, _, _, bot = service(tmp_path)
    calls = []

    async def polling(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(telegram.dispatcher, "start_polling", polling)
    asyncio.run(telegram.start_polling())

    assert calls == [((bot,), {"handle_signals": False})]


def add_search_run(database: Database) -> None:
    database.save_search_run(
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=12),
        state="completed",
        query_count=2,
        found_results=9,
        new_vacancies=4,
        duplicates=5,
        rejected_by_filter=2,
        rejected_by_llm=2,
        telegram_cards=0,
        error_count=1,
        rejection_reasons={"title": 2},
        error_reasons={"network_error": 1},
        last_safe_error="network_error",
        circuit_reason="page_structure_changed",
    )


def test_foreign_user_cannot_pause_or_read_configuration(tmp_path: Path) -> None:
    telegram, _, control, _ = service(tmp_path)

    reply = telegram.command("pause", user_id=99)

    assert reply == "This bot is private."
    assert not control.paused
    assert "42" not in reply
    assert "TG_USER_ID" not in reply


def test_pause_resume_and_status_use_live_state(tmp_path: Path) -> None:
    telegram, _, control, _ = service(tmp_path)

    assert telegram.command("pause", user_id=42) == "Agent paused."
    assert control.paused
    status = telegram.command("status", user_id=42)
    assert "mode: dry_run" in status
    assert "state: paused" in status
    assert telegram.command("resume", user_id=42) == "Agent resumed."
    assert not control.paused
    assert control.wake_event.is_set()


def test_diagnostics_shows_last_search_run(tmp_path: Path) -> None:
    telegram, database, control, _ = service(tmp_path)
    add_search_run(database)
    control.circuit_reason = "technical_failure_ratio"
    control.next_run_at = NOW + timedelta(minutes=30)

    reply = telegram.command("diagnostics", user_id=42)

    assert "длительность: 12 с" in reply
    assert "запросы: 2" in reply
    assert "найдено: 9" in reply
    assert "новые: 4" in reply
    assert "дубли: 5" in reply
    assert "фильтр: 2" in reply
    assert "LLM: 2" in reply
    assert "title=2" in reply
    assert "карточки: 0" in reply
    assert "ошибки: 1" in reply
    assert "network_error" in reply
    assert "breaker: open (technical_failure_ratio)" in reply
    assert "следующий запуск:" in reply


def test_resume_clears_runtime_breaker_but_keeps_saved_diagnostics(
    tmp_path: Path,
) -> None:
    telegram, database, control, _ = service(tmp_path)
    add_search_run(database)
    control.paused = True
    control.circuit_reason = "page_structure_changed"
    control.consecutive_search_errors = 3
    control.next_run_at = NOW + timedelta(minutes=30)

    assert telegram.command("resume", 42) == "Agent resumed."
    assert not control.paused
    assert control.circuit_reason == ""
    assert control.consecutive_search_errors == 0
    assert control.next_run_at is None
    assert control.wake_event.is_set()
    assert database.latest_search_run().circuit_reason == "page_structure_changed"
    diagnostics = telegram.command("diagnostics", 42)
    assert "breaker: closed" in diagnostics
    assert "last breaker: page_structure_changed" in diagnostics


def test_dry_run_preview_uses_html_with_company_details(tmp_path: Path) -> None:
    telegram, database, _, bot = service(tmp_path, "dry_run")
    add_preview_vacancy(database)

    asyncio.run(telegram.send_preview(database.get("job-1"), include_actions=False))

    assert len(bot.messages) == 1
    message = bot.messages[0]
    assert "Python &lt;Developer&gt;" in message["text"]
    assert "Example &amp; Co" in message["text"]
    assert "★ 4,7/5 · 128 отзывов" in message["text"]
    assert "<b>Опыт:</b> backend-сервисы." in message["text"]


def telegram_error(error_type: type[Exception], message: str) -> Exception:
    method = SendMessage(chat_id=42, text="test")
    return error_type(method=method, message=message)


def test_approval_preview_has_apply_and_manual_buttons(tmp_path: Path) -> None:
    telegram, database, _, bot = service(tmp_path, "approval")
    add_preview_vacancy(database)

    asyncio.run(telegram.send_preview(database.get("job-1"), include_actions=True))

    buttons = bot.messages[0]["reply_markup"].inline_keyboard[0]
    assert [button.callback_data for button in buttons] == [
        "apply:job-1", "manual:job-1"
    ]


def test_repeated_preview_updates_saved_message(tmp_path: Path) -> None:
    telegram, database, _, bot = service(tmp_path, "approval")
    add_preview_vacancy(database)

    async def scenario():
        vacancy = database.get("job-1")
        await telegram.send_preview(vacancy, include_actions=True)
        await telegram.send_preview(vacancy, include_actions=True)

    asyncio.run(scenario())

    assert len(bot.messages) == 1
    assert len(bot.edits) == 1
    assert bot.edits[0]["message_id"] == 1
    assert database.get_telegram_card("job-1").message_id == 1


def test_parallel_previews_create_only_one_message(tmp_path: Path) -> None:
    telegram, database, _, bot = service(tmp_path, "approval")
    add_preview_vacancy(database)

    async def scenario():
        vacancy = database.get("job-1")
        await asyncio.gather(*[
            telegram.send_preview(vacancy, include_actions=True)
            for _ in range(10)
        ])

    asyncio.run(scenario())

    assert len(bot.messages) == 1
    assert database.get_telegram_card("job-1").message_id == 1


def test_send_failure_releases_claim_for_retry(tmp_path: Path) -> None:
    telegram, database, _, bot = service(
        tmp_path,
        "approval",
        error=telegram_error(TelegramNetworkError, "network disconnected"),
    )
    add_preview_vacancy(database)

    async def scenario():
        vacancy = database.get("job-1")
        with pytest.raises(TelegramNetworkError):
            await telegram.send_preview(vacancy, include_actions=True)
        bot.error = None
        await telegram.send_preview(vacancy, include_actions=True)

    asyncio.run(scenario())

    assert len(bot.messages) == 1
    assert database.get_telegram_card("job-1").sending_at is None


def test_missing_telegram_message_is_resent_once(tmp_path: Path) -> None:
    telegram, database, _, bot = service(tmp_path, "approval")
    add_preview_vacancy(database)

    async def scenario():
        vacancy = database.get("job-1")
        await telegram.send_preview(vacancy, include_actions=True)
        bot.edit_error = telegram_error(
            TelegramBadRequest, "Bad Request: message to edit not found"
        )
        await telegram.send_preview(vacancy, include_actions=True)

    asyncio.run(scenario())

    assert len(bot.messages) == 2
    assert len(bot.edits) == 1
    assert database.get_telegram_card("job-1").message_id == 2


def test_same_title_on_different_job_ids_has_separate_cards(tmp_path: Path) -> None:
    telegram, database, _, bot = service(tmp_path, "approval")
    add_preview_vacancy(database)
    assert database.discover(
        job_id="job-2",
        title="Python <Developer>",
        company="Example & Co",
        company_url="https://hh.ru/employer/123",
        url="https://example.com/vacancy/job-2",
        description_hash="hash",
        search_query="Python",
        discovered_at=NOW,
    )
    assert database.request_approval(
        job_id="job-2",
        cover_letter="Hello <team>",
        llm_decision=True,
        llm_reason="Relevant & local",
        fit_summary="Опыт: backend-сервисы.",
        confidence=0.87,
        now=NOW,
    )

    async def scenario():
        await telegram.send_preview(database.get("job-1"), include_actions=True)
        await telegram.send_preview(database.get("job-2"), include_actions=True)

    asyncio.run(scenario())

    assert len(bot.messages) == 2
    assert database.get_telegram_card("job-1").message_id == 1
    assert database.get_telegram_card("job-2").message_id == 2


def test_queue_button_sends_page_link_instead_of_crashing(tmp_path: Path) -> None:
    telegram, database, _, bot = service(tmp_path, "approval")
    add_preview_vacancy(database)
    telegram.queue_page = SimpleNamespace(url=lambda: "http://127.0.0.1:8787/q/abc")
    answers: list[dict] = []

    async def answer(text: str, *, show_alert: bool = False) -> None:
        answers.append({"text": text, "show_alert": show_alert})

    callback = SimpleNamespace(
        from_user=SimpleNamespace(id=42), data="ctrl:pending",
        answer=answer, message=None,
    )

    asyncio.run(telegram._callback_handler(callback))

    assert bot.messages, "queue link message was not sent"
    button = bot.messages[0]["reply_markup"].inline_keyboard[0][0]
    assert button.url == "http://127.0.0.1:8787/q/abc"
    assert answers == [{"text": "Ссылка отправлена", "show_alert": False}]


def test_next_run_is_shown_in_schedule_timezone(tmp_path: Path) -> None:
    telegram, _, control, _ = service(tmp_path, "approval")
    telegram.settings = replace(
        telegram.settings,
        auto_apply=replace(telegram.settings.auto_apply, timezone="Europe/Moscow"),
    )
    control.next_run_at = datetime(2026, 11, 9, 5, 0, tzinfo=UTC)

    text, _ = telegram._menu_text_and_markup()

    assert "08:00 (Europe/Moscow)" in text
    assert "05:00" not in text


def test_search_command_forces_an_immediate_cycle(tmp_path: Path) -> None:
    telegram, _, control, _ = service(tmp_path, "approval")
    answers: list[str] = []

    async def answer(text: str, **kwargs) -> None:
        answers.append(text)

    message = SimpleNamespace(
        text="/search", from_user=SimpleNamespace(id=42), answer=answer
    )

    asyncio.run(telegram._command_handler(message))

    assert control.force_run is True
    assert control.wake_event.is_set()
    assert any("Прямо сейчас" in text or "прямо сейчас" in text for text in answers)


def test_search_button_forces_an_immediate_cycle(tmp_path: Path) -> None:
    telegram, _, control, _ = service(tmp_path, "approval")
    answers: list[str] = []

    async def answer(text: str, *, show_alert: bool = False) -> None:
        answers.append(text)

    callback = SimpleNamespace(
        from_user=SimpleNamespace(id=42), data="ctrl:search",
        answer=answer, message=None,
    )

    asyncio.run(telegram._callback_handler(callback))

    assert control.force_run is True
    assert control.wake_event.is_set()


def test_search_button_explains_an_exhausted_daily_limit(tmp_path: Path) -> None:
    telegram, database, control, _ = service(tmp_path, "approval")
    telegram.settings = replace(
        telegram.settings, max_applications_per_day=1
    )
    assert database.discover(
        job_id="job-applied",
        title="Analyst",
        company="Example",
        url="https://example.com/vacancy/job-applied",
        description_hash="hash",
        search_query="Analyst",
        discovered_at=NOW,
    )
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            "UPDATE vacancies SET status = 'applied', applied_at = ? WHERE id = ?",
            (NOW.isoformat(), "job-applied"),
        )
    answers: list[str] = []

    async def answer(text: str, *, show_alert: bool = False) -> None:
        answers.append(text)

    callback = SimpleNamespace(
        from_user=SimpleNamespace(id=42), data="ctrl:search",
        answer=answer, message=None,
    )

    asyncio.run(telegram._callback_handler(callback))

    assert control.force_run is False
    assert any("лимит" in text.lower() for text in answers)


def test_pending_and_stats_read_sqlite(tmp_path: Path) -> None:
    telegram, database, _, _ = service(tmp_path, "approval")
    add_preview_vacancy(database)

    assert "job-1" in telegram.command("pending", user_id=42)
    assert "pending_approval: 1" in telegram.command("stats", user_id=42)


@pytest.mark.parametrize("include_portfolio", [False, True])
def test_edit_cannot_remove_required_portfolio(tmp_path: Path, include_portfolio: bool) -> None:
    from types import SimpleNamespace
    telegram, database, _, bot = service(tmp_path, app_mode="approval")
    add_preview_vacancy(database)
    url = "https://portfolio.example/candidate"
    telegram.settings = replace(telegram.settings, profile=replace(
        telegram.settings.profile, cover_letter=replace(
            telegram.settings.profile.cover_letter, required_portfolio_url=url
        ),
    ))
    original = database.get("job-1").cover_letter
    edited = "My edited letter" + (f"\n\nПортфолио: {url}" if include_portfolio else "")
    async def answer(*args, **kwargs):
        pass
    callback = SimpleNamespace(from_user=SimpleNamespace(id=42), data="edit:job-1", answer=answer, message=None)
    async def scenario():
        task = asyncio.create_task(telegram._callback_handler(callback))
        await asyncio.sleep(0)
        assert telegram._edit_future is not None
        telegram._edit_future.set_result(edited)
        await task
    asyncio.run(scenario())
    assert database.get("job-1").cover_letter == (edited if include_portfolio else original)
    if not include_portfolio:
        assert any("не сохранено" in m["text"] for m in bot.messages)
    assert not bot.edits


def test_set_mode_auto_enables_hybrid_auto_apply(tmp_path: Path) -> None:
    telegram, _, _, _ = service(tmp_path, "approval")

    label, ok = telegram._set_mode("auto")

    assert ok is True
    assert "Auto-Apply" in label
    assert telegram.settings.app_mode == "auto"
    assert telegram.settings.enable_real_apply is True
    assert telegram.settings.auto_apply.enabled is True
    assert telegram._get_mode_name() == "auto"


def test_set_mode_approval_disables_auto_apply(tmp_path: Path) -> None:
    telegram, _, _, _ = service(tmp_path, "approval")
    telegram._set_mode("auto")

    _, ok = telegram._set_mode("approval")

    assert ok is True
    assert telegram.settings.app_mode == "approval"
    assert telegram.settings.auto_apply.enabled is False
    assert telegram._get_mode_name() == "approval"
