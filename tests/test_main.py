import asyncio
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import main as main_module
from ai_analyzer import AnalysisError, SuitabilityResult, VacancyAnalyzer
from approval import ApprovalResult
from config import load_settings
from database import Database, SearchRun, VacancyStatus
from hh_client import (
    CompanyDetails,
    PageState,
    VacancyDetails,
    VacancySearchResult,
    VacancySummary,
)
from llm.errors import LLMAuthenticationError, LLMTimeoutError
from llm.managed import ManagedLLMProvider
from llm.providers.fake import FakeProvider
from llm.types import LLMResponse
from main import (
    VacancyProcessResult,
    SearchRunStats,
    agent_loop,
    format_auto_batch_report,
    next_auto_batch_at,
    next_auto_window_start_after,
    planned_batch_size,
    process_vacancy,
    retry_failed_applications,
    restored_auto_batch_at,
    run_search_cycle,
)
from tests.test_config import VALID_ENV, write_profile
from tg_bot import AgentControl


NOW = datetime(2026, 7, 26, 9, 0, tzinfo=UTC)
SUMMARY = VacancySummary(
    "job-1", "Python developer", "https://example.com/vacancy/job-1", "Python"
)


class FakeHHClient:
    def __init__(
        self,
        details: VacancyDetails,
        search_results: list[VacancySearchResult] | None = None,
    ):
        self.details = details
        self.search_results = search_results or []
        self.read_calls = 0
        self.search_calls = 0
        self.message_checks = 0

    async def read_vacancy(self, summary, captcha_solver=None):
        self.read_calls += 1
        return self.details

    async def search_vacancies(self, query, areas, experience_filters, *, remote_only=False):
        self.search_calls += 1
        return (
            self.search_results.pop(0)
            if self.search_results
            else VacancySearchResult([], 0, 0)
        )

    async def check_messages(self, notify):
        self.message_checks += 1

    async def read_company_details(self, company_url: str):
        return CompanyDetails(4.7, 128)


class SearchReached(Exception):
    pass


class StopAtSearchHHClient(FakeHHClient):
    async def search_vacancies(self, *_args, **_kwargs):
        raise SearchReached


class FakeAnalyzer:
    def __init__(self):
        self.assess_calls = 0

    async def assess(self, title: str, description: str) -> SuitabilityResult:
        self.assess_calls += 1
        return SuitabilityResult(
            suitable=True,
            confidence=0.91,
            reason="Relevant work",
            fit_points=[{"category": "Навыки", "text": "Python"}],
        )

    async def generate_cover_letter(
        self, title: str, description: str, company_name: str = ""
    ) -> str:
        return "Safe local-profile letter"


class SequenceAnalyzer(FakeAnalyzer):
    def __init__(self, outcomes: list[SuitabilityResult | Exception]):
        self.outcomes = iter(outcomes)

    async def assess(self, title: str, description: str) -> SuitabilityResult:
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class PausingAnalyzer(FakeAnalyzer):
    def __init__(self, control: AgentControl):
        super().__init__()
        self.control = control
        self.calls = 0

    async def assess(self, title: str, description: str) -> SuitabilityResult:
        self.calls += 1
        self.control.paused = True
        return await super().assess(title, description)


class UncertainApplicationService:
    """Emulates a submit whose HH confirmation never arrived."""

    def __init__(self, database: Database, app_settings):
        self.database = database
        self.settings = app_settings
        self.calls: list[str] = []

    async def auto_apply(self, job_id: str) -> ApprovalResult:
        self.calls.append(job_id)
        permit = self.database.approve(
            job_id, 42, 42, NOW, self.settings.approval_ttl_minutes
        )
        assert permit
        claim = self.database.claim_application(
            job_id=job_id,
            permit=permit,
            telegram_user_id=42,
            expected_user_id=42,
            app_mode=self.settings.app_mode,
            enable_real_apply=True,
            daily_limit=self.settings.max_applications_per_day,
            now=NOW,
        )
        assert claim.allowed
        assert self.database.mark_submit_attempt(
            job_id,
            permit,
            now=NOW,
            daily_limit=self.settings.max_applications_per_day,
        )
        assert self.database.complete_application(
            job_id,
            permit,
            success=False,
            now=NOW,
            error_text="HH.ru did not confirm the application",
        )
        return ApprovalResult(False, "Application failed")


class FakeTelegram:
    def __init__(self):
        self.previews: list[tuple[str, bool]] = []
        self.notifications: list[str] = []
        self.screenshots: list[tuple[object, str]] = []

    async def notify_screenshot(self, path, caption: str) -> None:
        self.screenshots.append((path, caption))

    async def send_preview(self, vacancy, include_actions: bool):
        self.previews.append((vacancy.id, include_actions))

    async def notify(self, text: str):
        self.notifications.append(text)

    async def notify_analysis_failed(self, title: str, url: str, error_type: str):
        self.notifications.append(f"analysis_failed:{error_type}")

    async def request_captcha(self, screenshot, title: str, timeout_seconds: int):
        return None


class FakeAutoApprovalService:
    def __init__(self, outcome: ApprovalResult | None = None):
        self.outcome = outcome or ApprovalResult(True, "Application sent")
        self.calls: list[str] = []

    async def auto_apply(self, job_id: str) -> ApprovalResult:
        self.calls.append(job_id)
        return self.outcome


def settings(tmp_path: Path, mode: str):
    loaded = load_settings(
        profile_path=write_profile(tmp_path),
        environ={**VALID_ENV, "TG_USER_ID": "42", "APP_MODE": mode},
    )
    return replace(loaded, database_path=tmp_path / "agent.db")


def auto_settings(tmp_path: Path):
    loaded = settings(tmp_path, "approval")
    return replace(
        loaded,
        max_applications_per_day=20,
        auto_apply=replace(
            loaded.auto_apply,
            enabled=True,
            min_confidence=0.85,
            min_batch_size=5,
            max_batch_size=6,
            min_interval_hours=3,
            max_interval_hours=5,
            start_hour=10,
            end_hour=22,
            timezone="Europe/Berlin",
        ),
    )


def test_run_shares_mistral_manager_and_assigns_telegram_notifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app_settings = settings(tmp_path, "approval")
    captured: dict[str, object] = {}

    class FakeBackend:
        async def start(self):
            return object()

        async def close(self):
            return None

    class FakeMistralManager:
        def __init__(self):
            self.notifier = None

        def set_notifier(self, notifier):
            self.notifier = notifier

        async def close(self):
            return None

    manager = FakeMistralManager()

    class FakeHHClient:
        def __init__(self, *_args):
            pass

        async def ensure_login(self) -> bool:
            return True

    class FakeAnalyzer:
        def __init__(self, _settings, provider):
            captured["analyzer_provider"] = provider

    class FakeTelegramService:
        def __init__(
            self,
            _settings,
            _database,
            _approval_service,
            _control,
            *,
            mistral_keys,
            **_kwargs,
        ):
            captured["telegram"] = self
            captured["telegram_manager"] = mistral_keys

        async def notify(self, _text: str) -> None:
            return None

        async def check_updates(self, notify: bool = True) -> None:
            return None

        async def start_polling(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    async def fake_agent_loop(*_args) -> None:
        return None

    async def fake_resume_bumper_loop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(
        main_module, "resume_bumper_loop", fake_resume_bumper_loop
    )
    monkeypatch.setattr(main_module, "MistralKeyManager", FakeMistralManager, raising=False)
    monkeypatch.setattr(main_module, "create_browser_backend", lambda _settings: FakeBackend())
    monkeypatch.setattr(main_module, "create_llm_provider", lambda *_args: manager)
    monkeypatch.setattr(main_module, "HHClient", FakeHHClient)
    monkeypatch.setattr(main_module, "VacancyAnalyzer", FakeAnalyzer)
    monkeypatch.setattr(main_module, "TelegramService", FakeTelegramService)
    monkeypatch.setattr(main_module, "agent_loop", fake_agent_loop)

    asyncio.run(main_module.run(app_settings))

    telegram = captured["telegram"]
    assert captured["analyzer_provider"] is manager
    assert captured["telegram_manager"] is manager
    assert manager.notifier == telegram.notify


def test_dry_run_records_and_previews_without_pending_actions(tmp_path: Path) -> None:
    app_settings = settings(tmp_path, "dry_run")
    database = Database(app_settings.database_path)
    database.init()
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )
    class CompanyCheckingAnalyzer(FakeAnalyzer):
        async def generate_cover_letter(self, title, description, company_name=""):
            assert company_name == "Example"
            return await super().generate_cover_letter(title, description, company_name)

    asyncio.run(
        process_vacancy(
            SUMMARY,
            app_settings,
            database,
            FakeHHClient(details),
            CompanyCheckingAnalyzer(),
            telegram,
            now_factory=lambda: NOW,
        )
    )

    vacancy = database.get("job-1")
    assert vacancy.status is VacancyStatus.DISCOVERED
    assert vacancy.cover_letter == "Safe local-profile letter"
    assert vacancy.fit_summary.startswith("Навыки: Python")
    assert vacancy.company_rating == 4.7
    assert telegram.previews == [("job-1", False)]


def test_approval_mode_records_pending_and_sends_actions(tmp_path: Path) -> None:
    app_settings = settings(tmp_path, "approval")
    database = Database(app_settings.database_path)
    database.init()
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )

    asyncio.run(
        process_vacancy(
            SUMMARY,
            app_settings,
            database,
            FakeHHClient(details),
            FakeAnalyzer(),
            telegram,
            now_factory=lambda: NOW,
        )
    )

    assert database.get("job-1").status is VacancyStatus.PENDING_APPROVAL
    assert telegram.previews == [("job-1", True)]


def test_auto_apply_uses_only_high_confidence_vacancies(tmp_path: Path) -> None:
    app_settings = auto_settings(tmp_path)
    database = Database(app_settings.database_path)
    database.init()
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )
    auto_service = FakeAutoApprovalService()

    result = asyncio.run(
        process_vacancy(
            SUMMARY,
            app_settings,
            database,
            FakeHHClient(details),
            FakeAnalyzer(),
            telegram,
            approval_service=auto_service,
            now_factory=lambda: NOW,
        )
    )

    assert result.outcome == "auto_applied"
    assert auto_service.calls == [SUMMARY.id]
    assert telegram.previews == []
    assert len(telegram.notifications) == 1
    assert telegram.notifications[0].startswith(
        "✓ Автоотклик отправлен: Python developer"
    )
    assert SUMMARY.url in telegram.notifications[0]
    assert "Example" in telegram.notifications[0]


def test_auto_apply_keeps_lower_confidence_vacancy_for_manual_review(
    tmp_path: Path,
) -> None:
    app_settings = auto_settings(tmp_path)
    database = Database(app_settings.database_path)
    database.init()
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )
    auto_service = FakeAutoApprovalService()
    analyzer = SequenceAnalyzer(
        [SuitabilityResult(suitable=True, confidence=0.84, reason="Relevant work")]
    )

    result = asyncio.run(
        process_vacancy(
            SUMMARY,
            app_settings,
            database,
            FakeHHClient(details),
            analyzer,
            telegram,
            approval_service=auto_service,
            now_factory=lambda: NOW,
        )
    )

    assert result.outcome == "telegram_card"
    assert auto_service.calls == []
    assert database.get(SUMMARY.id).status is VacancyStatus.PENDING_APPROVAL
    assert telegram.previews == [(SUMMARY.id, True)]


def auto_mode_settings(tmp_path: Path):
    loaded = load_settings(
        profile_path=write_profile(tmp_path),
        environ={
            **VALID_ENV,
            "TG_USER_ID": "42",
            "APP_MODE": "auto",
            "ENABLE_REAL_APPLY": "true",
            "MAX_APPLICATIONS_PER_DAY": "20",
        },
    )
    return replace(
        loaded,
        database_path=tmp_path / "agent.db",
        auto_apply=replace(
            loaded.auto_apply,
            enabled=True,
            min_confidence=0.85,
            min_batch_size=5,
            max_batch_size=6,
            min_interval_hours=3,
            max_interval_hours=5,
            start_hour=0,
            end_hour=23,
            timezone="UTC",
        ),
    )


def test_auto_mode_applies_to_non_ai_vacancy_without_confidence_gate(
    tmp_path: Path,
) -> None:
    app_settings = auto_mode_settings(tmp_path)
    database = Database(app_settings.database_path)
    database.init()
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )
    auto_service = FakeAutoApprovalService()
    analyzer = SequenceAnalyzer(
        [SuitabilityResult(suitable=True, confidence=0.4, reason="Relevant work")]
    )

    result = asyncio.run(
        process_vacancy(
            SUMMARY,
            app_settings,
            database,
            FakeHHClient(details),
            analyzer,
            telegram,
            approval_service=auto_service,
            now_factory=lambda: NOW,
        )
    )

    assert result.outcome == "auto_applied"
    assert auto_service.calls == [SUMMARY.id]
    assert telegram.previews == []


def test_auto_mode_keeps_ai_vacancy_for_manual_telegram_reply(
    tmp_path: Path,
) -> None:
    app_settings = auto_mode_settings(tmp_path)
    database = Database(app_settings.database_path)
    database.init()
    database.set_manual_review_disabled(False)
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY,
        PageState.VACANCY_LOADED,
        "Example",
        "Build LLM agents with RAG",
    )
    auto_service = FakeAutoApprovalService()
    analyzer = SequenceAnalyzer(
        [
            SuitabilityResult(
                suitable=True,
                confidence=0.99,
                reason="AI priority vacancy",
                is_ai_priority=True,
            )
        ]
    )

    result = asyncio.run(
        process_vacancy(
            SUMMARY,
            app_settings,
            database,
            FakeHHClient(details),
            analyzer,
            telegram,
            approval_service=auto_service,
            now_factory=lambda: NOW,
        )
    )

    vacancy = database.get(SUMMARY.id)
    assert result.outcome == "telegram_card"
    assert auto_service.calls == []
    assert telegram.previews == [(SUMMARY.id, True)]
    assert vacancy is not None and vacancy.is_ai_priority is True
    assert vacancy.status is VacancyStatus.PENDING_APPROVAL


def test_ai_vacancy_is_applied_automatically_when_manual_work_is_off(
    tmp_path: Path,
) -> None:
    app_settings = auto_mode_settings(tmp_path)
    database = Database(app_settings.database_path)
    database.init()
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY,
        PageState.VACANCY_LOADED,
        "Example",
        "Build LLM agents with RAG",
    )
    auto_service = FakeAutoApprovalService()
    analyzer = SequenceAnalyzer(
        [
            SuitabilityResult(
                suitable=True,
                confidence=0.99,
                reason="AI priority vacancy",
                is_ai_priority=True,
            )
        ]
    )

    result = asyncio.run(
        process_vacancy(
            SUMMARY,
            app_settings,
            database,
            FakeHHClient(details),
            analyzer,
            telegram,
            approval_service=auto_service,
            now_factory=lambda: NOW,
        )
    )

    vacancy = database.get(SUMMARY.id)
    assert result.outcome == "auto_applied"
    assert auto_service.calls == [SUMMARY.id]
    assert telegram.previews == []
    assert vacancy is not None and vacancy.is_ai_priority is True


def test_ai_vacancy_outside_the_analyst_role_is_still_rejected(
    tmp_path: Path,
) -> None:
    app_settings = auto_mode_settings(tmp_path)
    database = Database(app_settings.database_path)
    database.init()
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY,
        PageState.VACANCY_LOADED,
        "Example",
        "Ищем менеджера, который поможет внедрить ИИ в поддержку клиентов",
    )
    auto_service = FakeAutoApprovalService()
    analyzer = SequenceAnalyzer(
        [
            SuitabilityResult(
                suitable=False,
                confidence=0.4,
                reason="Роль не аналитическая",
            )
        ]
    )

    result = asyncio.run(
        process_vacancy(
            SUMMARY,
            app_settings,
            database,
            FakeHHClient(details),
            analyzer,
            telegram,
            approval_service=auto_service,
            now_factory=lambda: NOW,
        )
    )

    vacancy = database.get(SUMMARY.id)
    assert result.outcome == "rejected_by_llm"
    assert auto_service.calls == []
    assert telegram.previews == []
    assert vacancy is not None and vacancy.status is VacancyStatus.REJECTED_BY_LLM


def test_auto_batch_stops_after_its_send_target(tmp_path: Path) -> None:
    app_settings = auto_settings(tmp_path)
    database = Database(app_settings.database_path)
    database.init()
    telegram = FakeTelegram()
    summaries = [
        SUMMARY,
        VacancySummary("job-2", "Backend developer", "https://example.com/vacancy/job-2", "Python"),
    ]
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )
    auto_service = FakeAutoApprovalService()

    run = asyncio.run(
        run_search_cycle(
            app_settings,
            database,
            FakeHHClient(details, [VacancySearchResult(summaries, 2, 0)]),
            FakeAnalyzer(),
            telegram,
            AgentControl(),
            approval_service=auto_service,
            auto_apply_batch_limit=1,
            now_factory=lambda: NOW,
        )
    )

    assert auto_service.calls == [SUMMARY.id]
    assert run.new_vacancies == 2
    assert database.get("job-2") is None


def test_single_unconfirmed_delivery_does_not_pause_the_batch(
    tmp_path: Path,
) -> None:
    app_settings = auto_settings(tmp_path)
    database = Database(app_settings.database_path)
    database.init()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )
    service = UncertainApplicationService(database, app_settings)
    control = AgentControl()

    run = asyncio.run(
        run_search_cycle(
            app_settings,
            database,
            FakeHHClient(details, [VacancySearchResult([SUMMARY], 1, 0)]),
            FakeAnalyzer(),
            FakeTelegram(),
            control,
            approval_service=service,
            now_factory=lambda: NOW,
        )
    )

    assert service.calls == [SUMMARY.id]
    assert not control.paused
    assert run.state == "completed"
    vacancy = database.get(SUMMARY.id)
    assert vacancy.status is VacancyStatus.RETRY_PENDING
    assert vacancy.apply_retry_count == 1


def test_failed_application_sends_screen_snapshot(tmp_path: Path) -> None:
    app_settings = auto_settings(tmp_path)
    database = Database(app_settings.database_path)
    database.init()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )
    hh_client = FakeHHClient(details)
    hh_client.last_failure_screenshot = tmp_path / "failure.png"
    telegram = FakeTelegram()
    service = UncertainApplicationService(database, app_settings)

    result = asyncio.run(
        process_vacancy(
            SUMMARY,
            app_settings,
            database,
            hh_client,
            FakeAnalyzer(),
            telegram,
            approval_service=service,
            now_factory=lambda: NOW,
        )
    )

    vacancy = database.get(SUMMARY.id)
    assert result.outcome == "retry_scheduled"
    assert vacancy.status is VacancyStatus.RETRY_PENDING
    assert vacancy.apply_retry_count == 1
    assert telegram.screenshots == []
    assert any("повторю в следующем цикле" in text for text in telegram.notifications)


def test_second_failure_parks_the_vacancy_in_the_queue(tmp_path: Path) -> None:
    app_settings = auto_settings(tmp_path)
    database = Database(app_settings.database_path)
    database.init()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )
    telegram = FakeTelegram()
    service = UncertainApplicationService(database, app_settings)
    asyncio.run(
        process_vacancy(
            SUMMARY,
            app_settings,
            database,
            FakeHHClient(details),
            FakeAnalyzer(),
            telegram,
            approval_service=service,
            now_factory=lambda: NOW,
        )
    )

    asyncio.run(retry_failed_applications(database, service, telegram, 5))

    vacancy = database.get(SUMMARY.id)
    assert vacancy is not None
    assert vacancy.apply_retry_count == 2
    assert vacancy.status is VacancyStatus.PENDING_APPROVAL
    assert telegram.previews == [(SUMMARY.id, True)]


def test_repeated_unconfirmed_deliveries_pause_the_agent(tmp_path: Path) -> None:
    app_settings = auto_settings(tmp_path)
    database = Database(app_settings.database_path)
    database.init()
    second = VacancySummary(
        "job-2", "Backend developer", "https://example.com/vacancy/job-2", "Python"
    )
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )
    service = UncertainApplicationService(database, app_settings)
    control = AgentControl()

    run = asyncio.run(
        run_search_cycle(
            app_settings,
            database,
            FakeHHClient(details, [VacancySearchResult([SUMMARY, second], 2, 0)]),
            FakeAnalyzer(),
            FakeTelegram(),
            control,
            approval_service=service,
            now_factory=lambda: NOW,
        )
    )

    assert service.calls == [SUMMARY.id, second.id]
    assert control.paused
    assert control.circuit_reason == "application_delivery_uncertain"
    assert run.state == "paused_by_circuit_breaker"


def test_auto_batch_schedule_stays_inside_configured_daytime_window(
    tmp_path: Path,
) -> None:
    app_settings = auto_settings(tmp_path)
    before_window = datetime(2026, 7, 26, 7, tzinfo=UTC)
    late_window = datetime(2026, 7, 26, 21, tzinfo=UTC)

    first = next_auto_batch_at(before_window, app_settings)
    after_late_batch = next_auto_batch_at(
        late_window, app_settings, interval_hours=5
    )
    after_limit = next_auto_window_start_after(late_window, app_settings)

    local_timezone = ZoneInfo("Europe/Berlin")
    assert first.astimezone(local_timezone).hour == 10
    assert after_late_batch.astimezone(local_timezone).hour == 10
    assert after_late_batch.date() > late_window.astimezone(local_timezone).date()
    assert after_limit.astimezone(local_timezone).hour == 10


def test_auto_schedule_restart_preserves_minimum_interval(tmp_path: Path) -> None:
    app_settings = auto_settings(tmp_path)
    database = Database(app_settings.database_path)
    database.init()
    last_batch_finished = datetime(2026, 7, 26, 5, tzinfo=UTC)
    database.save_search_run(
        started_at=last_batch_finished - timedelta(minutes=5),
        finished_at=last_batch_finished,
        state="completed",
        query_count=1,
        found_results=0,
        new_vacancies=0,
        duplicates=0,
        rejected_by_filter=0,
        rejected_by_llm=0,
        telegram_cards=0,
        error_count=0,
        rejection_reasons={},
        error_reasons={},
    )

    restart_at = last_batch_finished + timedelta(minutes=10)
    next_batch = restored_auto_batch_at(
        restart_at, app_settings, database, interval_hours=3
    )

    assert next_batch == last_batch_finished + timedelta(hours=3)


def test_auto_schedule_skips_stale_run_into_the_future(tmp_path: Path) -> None:
    app_settings = auto_settings(tmp_path)
    database = Database(app_settings.database_path)
    database.init()
    stale_finished = datetime(2026, 7, 20, 5, tzinfo=UTC)
    database.save_search_run(
        started_at=stale_finished - timedelta(minutes=5),
        finished_at=stale_finished,
        state="completed",
        query_count=1,
        found_results=0,
        new_vacancies=0,
        duplicates=0,
        rejected_by_filter=0,
        rejected_by_llm=0,
        telegram_cards=0,
        error_count=0,
        rejection_reasons={},
        error_reasons={},
    )

    restart_at = datetime(2026, 7, 26, 7, tzinfo=UTC)
    next_batch = restored_auto_batch_at(
        restart_at, app_settings, database, interval_hours=3
    )

    assert next_batch > restart_at
    assert next_batch.astimezone(ZoneInfo("Europe/Berlin")).hour == 10
    assert next_batch.astimezone(ZoneInfo("Europe/Berlin")).date() == restart_at.astimezone(
        ZoneInfo("Europe/Berlin")
    ).date()


def test_auto_batch_report_contains_result_and_next_run(tmp_path: Path) -> None:
    app_settings = auto_settings(tmp_path)
    run = SearchRun(
        id=1,
        started_at="2026-07-26T05:00:00+00:00",
        finished_at="2026-07-26T05:10:00+00:00",
        state="completed",
        query_count=3,
        found_results=60,
        new_vacancies=12,
        duplicates=4,
        rejected_by_filter=5,
        rejected_by_llm=3,
        telegram_cards=2,
        error_count=1,
        rejection_reasons={},
        error_reasons={},
        last_safe_error="",
        circuit_reason="",
    )

    report = format_auto_batch_report(
        run,
        auto_applied=4,
        next_batch=datetime(2026, 7, 26, 8, 10, tzinfo=UTC),
        settings=app_settings,
    )

    assert "Найдено: 60; новых: 12." in report
    assert "Отправлено: 4; на ручную проверку: 2." in report
    assert "Отсечено фильтрами: 8; безопасно остановлено: 1." in report
    assert "Следующая пачка: 26.07 10:10 Europe/Berlin." in report


def test_browser_read_error_is_persisted_as_apply_failed(tmp_path: Path) -> None:
    app_settings = settings(tmp_path, "dry_run")
    database = Database(app_settings.database_path)
    database.init()
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY, PageState.NETWORK_ERROR, error="navigation failed"
    )

    asyncio.run(
        process_vacancy(
            SUMMARY,
            app_settings,
            database,
            FakeHHClient(details),
            FakeAnalyzer(),
            telegram,
            now_factory=lambda: NOW,
        )
    )

    vacancy = database.get("job-1")
    assert vacancy.status is VacancyStatus.APPLY_FAILED
    assert "navigation failed" in vacancy.error_text


def test_already_responded_vacancy_is_skipped_without_llm_work(
    tmp_path: Path,
) -> None:
    app_settings = settings(tmp_path, "dry_run")
    database = Database(app_settings.database_path)
    database.init()
    telegram = FakeTelegram()
    analyzer = FakeAnalyzer()
    details = VacancyDetails(SUMMARY, PageState.ALREADY_RESPONDED)

    result = asyncio.run(
        process_vacancy(
            SUMMARY,
            app_settings,
            database,
            FakeHHClient(details),
            analyzer,
            telegram,
            now_factory=lambda: NOW,
        )
    )

    vacancy = database.get("job-1")
    assert result.outcome == "skipped"
    assert result.reason == "already_responded"
    assert analyzer.assess_calls == 0
    assert telegram.previews == []
    assert vacancy.status is VacancyStatus.SKIPPED
    assert vacancy.error_text == "already_responded"


def test_llm_failure_records_analysis_failure_without_requesting_approval(
    tmp_path: Path,
) -> None:
    app_settings = settings(tmp_path, "approval")
    database = Database(app_settings.database_path)
    database.init()
    provider = ManagedLLMProvider(
        FakeProvider([LLMAuthenticationError()]),
        database,
        max_retries=1,
        max_requests_per_day=100,
    )
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )

    asyncio.run(
        process_vacancy(
            SUMMARY,
            app_settings,
            database,
            FakeHHClient(details),
            VacancyAnalyzer(app_settings, provider),
            telegram,
            now_factory=lambda: NOW,
        )
    )

    vacancy = database.get("job-1")
    assert vacancy.status is VacancyStatus.ANALYSIS_FAILED
    assert vacancy.llm_decision is None
    assert vacancy.error_text == "authentication"
    assert vacancy.analysis_retry_count == 1
    assert vacancy.analysis_next_retry_at == (
        NOW + timedelta(minutes=app_settings.check_interval_minutes)
    ).isoformat()
    assert vacancy.cover_letter == ""
    assert telegram.previews == []
    assert telegram.notifications == ["analysis_failed:authentication"]


def test_analysis_retry_delay_starts_when_failed_analysis_finishes(
    tmp_path: Path,
) -> None:
    app_settings = settings(tmp_path, "approval")
    database = Database(app_settings.database_path)
    database.init()
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )
    times = iter((NOW, NOW + timedelta(minutes=5)))

    asyncio.run(
        process_vacancy(
            SUMMARY,
            app_settings,
            database,
            FakeHHClient(details),
            SequenceAnalyzer([AnalysisError("timeout")]),
            telegram,
            now_factory=lambda: next(times),
        )
    )

    assert database.get(SUMMARY.id).analysis_next_retry_at == (
        NOW
        + timedelta(minutes=5)
        + timedelta(minutes=app_settings.check_interval_minutes)
    ).isoformat()


def test_failed_retry_delay_starts_when_retry_finishes(tmp_path: Path) -> None:
    app_settings = settings(tmp_path, "approval")
    database = Database(app_settings.database_path)
    database.init()
    assert database.discover(
        job_id=SUMMARY.id,
        title=SUMMARY.title,
        company="Example",
        url=SUMMARY.url,
        description_hash="abc123",
        search_query=SUMMARY.search_query,
        discovered_at=NOW,
    )
    assert database.mark_analysis_failed(
        SUMMARY.id,
        error_type="timeout",
        now=NOW,
        retry_after=timedelta(minutes=30),
        max_attempts=3,
    )
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )
    times = iter(
        (
            NOW + timedelta(minutes=30),
            NOW + timedelta(minutes=30),
            NOW + timedelta(minutes=35),
        )
    )

    asyncio.run(
        main_module.retry_due_analyses(
            app_settings,
            database,
            FakeHHClient(details),
            SequenceAnalyzer([AnalysisError("timeout")]),
            telegram,
            AgentControl(),
            now_factory=lambda: next(times),
        )
    )

    vacancy = database.get(SUMMARY.id)
    assert vacancy.analysis_retry_count == 2
    assert vacancy.analysis_next_retry_at == (
        NOW
        + timedelta(minutes=35)
        + timedelta(minutes=app_settings.check_interval_minutes)
    ).isoformat()


def test_pause_stops_retry_backlog_before_next_vacancy(tmp_path: Path) -> None:
    app_settings = settings(tmp_path, "approval")
    database = Database(app_settings.database_path)
    database.init()
    second = VacancySummary(
        "job-2", "Backend developer", "https://example.com/vacancy/job-2", "Python"
    )
    for offset, summary in enumerate((SUMMARY, second)):
        discovered_at = NOW + timedelta(seconds=offset)
        assert database.discover(
            job_id=summary.id,
            title=summary.title,
            company="Example",
            url=summary.url,
            description_hash=f"hash-{offset}",
            search_query=summary.search_query,
            discovered_at=discovered_at,
        )
        assert database.mark_analysis_failed(
            summary.id,
            error_type="timeout",
            now=discovered_at,
            retry_after=timedelta(minutes=30),
            max_attempts=3,
        )
    control = AgentControl()
    analyzer = PausingAnalyzer(control)
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )

    asyncio.run(
        main_module.retry_due_analyses(
            app_settings,
            database,
            FakeHHClient(details),
            analyzer,
            FakeTelegram(),
            control,
            now_factory=lambda: NOW + timedelta(minutes=31),
        )
    )

    assert analyzer.calls == 1
    assert database.get(SUMMARY.id).status is VacancyStatus.PENDING_APPROVAL
    assert database.get(second.id).status is VacancyStatus.DISCOVERED


def test_three_timeouts_are_retried_in_later_cycle(tmp_path: Path) -> None:
    app_settings = settings(tmp_path, "approval")
    database = Database(app_settings.database_path)
    database.init()
    adapter = FakeProvider(
        [
            LLMTimeoutError(),
            LLMTimeoutError(),
            LLMTimeoutError(),
            LLMResponse(
                text='{"suitable":true,"confidence":0.91,"reason":"Relevant work"}',
                provider="fake",
                model="test-model",
            ),
            LLMResponse(
                text="Safe local-profile letter", provider="fake", model="test-model"
            ),
        ]
    )

    async def no_sleep(_seconds: float) -> None:
        pass

    provider = ManagedLLMProvider(
        adapter,
        database,
        max_retries=2,
        max_requests_per_day=100,
        sleep=no_sleep,
        now_factory=lambda: NOW,
    )
    analyzer = VacancyAnalyzer(app_settings, provider)
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )

    asyncio.run(
        process_vacancy(
            SUMMARY,
            app_settings,
            database,
            FakeHHClient(details),
            analyzer,
            telegram,
            now_factory=lambda: NOW,
        )
    )
    assert database.get(SUMMARY.id).status is VacancyStatus.ANALYSIS_FAILED

    asyncio.run(
        main_module.retry_due_analyses(
            app_settings,
            database,
            FakeHHClient(details),
            analyzer,
            telegram,
            AgentControl(),
            now_factory=lambda: NOW + timedelta(minutes=30),
        )
    )

    vacancy = database.get(SUMMARY.id)
    assert vacancy.status is VacancyStatus.PENDING_APPROVAL
    assert vacancy.llm_decision is True
    assert len(adapter.requests) == 5


def test_manual_search_runs_even_outside_the_window(tmp_path: Path) -> None:
    app_settings = auto_settings(tmp_path)
    database = Database(app_settings.database_path)
    database.init()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )
    control = AgentControl()
    control.force_run = True
    outside_window = datetime(2026, 11, 9, 0, 0, tzinfo=UTC)  # 03:00 MSK

    with pytest.raises(SearchReached):
        asyncio.run(
            agent_loop(
                app_settings,
                database,
                StopAtSearchHHClient(details),
                FakeAnalyzer(),
                FakeTelegram(),
                control,
                now_factory=lambda: outside_window,
            )
        )

    assert control.force_run is False


def test_agent_loop_retries_due_analysis_before_new_search(tmp_path: Path) -> None:
    app_settings = settings(tmp_path, "approval")
    database = Database(app_settings.database_path)
    database.init()
    assert database.discover(
        job_id=SUMMARY.id,
        title=SUMMARY.title,
        company="Example",
        url=SUMMARY.url,
        description_hash="abc123",
        search_query=SUMMARY.search_query,
        discovered_at=NOW,
    )
    assert database.mark_analysis_failed(
        SUMMARY.id,
        error_type="timeout",
        now=NOW,
        retry_after=timedelta(minutes=30),
        max_attempts=3,
    )
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )

    with pytest.raises(SearchReached):
        asyncio.run(
            agent_loop(
                app_settings,
                database,
                StopAtSearchHHClient(details),
                SequenceAnalyzer(
                    [
                        SuitabilityResult(
                            suitable=True,
                            confidence=0.91,
                            reason="Relevant work",
                        )
                    ]
                ),
                telegram,
                AgentControl(),
            )
        )

    vacancy = database.get(SUMMARY.id)
    assert vacancy.status is VacancyStatus.PENDING_APPROVAL
    assert vacancy.error_text == ""
    assert telegram.previews == [(SUMMARY.id, True)]


def test_agent_loop_resumes_interrupted_discovered_analysis_before_search(
    tmp_path: Path,
) -> None:
    app_settings = settings(tmp_path, "approval")
    database = Database(app_settings.database_path)
    database.init()
    assert database.discover(
        job_id=SUMMARY.id,
        title=SUMMARY.title,
        company="Example",
        url=SUMMARY.url,
        description_hash="abc123",
        search_query=SUMMARY.search_query,
        discovered_at=NOW,
    )
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )

    with pytest.raises(SearchReached):
        asyncio.run(
            agent_loop(
                app_settings,
                database,
                StopAtSearchHHClient(details),
                SequenceAnalyzer(
                    [
                        SuitabilityResult(
                            suitable=True,
                            confidence=0.91,
                            reason="Relevant work",
                        )
                    ]
                ),
                telegram,
                AgentControl(),
            )
        )

    assert database.get(SUMMARY.id).status is VacancyStatus.PENDING_APPROVAL
    assert telegram.previews == [(SUMMARY.id, True)]


def test_retry_uses_valid_negative_decision_instead_of_technical_error(
    tmp_path: Path,
) -> None:
    app_settings = settings(tmp_path, "approval")
    database = Database(app_settings.database_path)
    database.init()
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )
    analyzer = SequenceAnalyzer(
        [
            AnalysisError("timeout"),
            SuitabilityResult(
                suitable=False,
                confidence=0.2,
                reason="Role mismatch",
            ),
        ]
    )

    asyncio.run(
        process_vacancy(
            SUMMARY,
            app_settings,
            database,
            FakeHHClient(details),
            analyzer,
            telegram,
            now_factory=lambda: NOW,
        )
    )
    asyncio.run(
        main_module.retry_due_analyses(
            app_settings,
            database,
            FakeHHClient(details),
            analyzer,
            telegram,
            AgentControl(),
            now_factory=lambda: NOW + timedelta(minutes=30),
        )
    )

    vacancy = database.get(SUMMARY.id)
    assert vacancy.status is VacancyStatus.REJECTED_BY_LLM
    assert vacancy.llm_decision is False
    assert vacancy.llm_reason == "Role mismatch"
    assert vacancy.error_text == ""
    assert telegram.previews == []


def test_pending_search_duplicate_does_not_send_or_change_status(
    tmp_path: Path,
) -> None:
    app_settings = settings(tmp_path, "approval")
    database = Database(app_settings.database_path)
    database.init()
    assert database.discover(
        job_id=SUMMARY.id,
        title=SUMMARY.title,
        company="Example",
        company_url="",
        url=SUMMARY.url,
        description_hash="hash",
        search_query=SUMMARY.search_query,
        discovered_at=NOW,
    )
    assert database.request_approval(
        job_id=SUMMARY.id,
        cover_letter="Letter",
        llm_decision=True,
        llm_reason="Relevant",
        fit_summary="Опыт: Python.",
        confidence=0.9,
        now=NOW,
    )
    telegram = FakeTelegram()
    hh_client = FakeHHClient(
        VacancyDetails(SUMMARY, PageState.VACANCY_LOADED, "Example", "Description")
    )

    result = asyncio.run(
        process_vacancy(
            replace(SUMMARY, previously_sent=True),
            app_settings,
            database,
            hh_client,
            FakeAnalyzer(),
            telegram,
            now_factory=lambda: NOW,
        )
    )

    assert result == VacancyProcessResult("ignored")
    assert database.get(SUMMARY.id).status is VacancyStatus.PENDING_APPROVAL
    assert telegram.previews == []
    assert hh_client.read_calls == 0


def test_cli_reports_configuration_error_without_traceback(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--check-config",
            "--env-file",
            str(tmp_path / "missing.env"),
            "--profile",
            str(tmp_path / "missing.yaml"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stderr.startswith("Configuration error:")
    assert "Traceback" not in result.stderr


def _quota_settings(tmp_path: Path, daily_limit: int = 120):
    loaded = auto_mode_settings(tmp_path)
    return replace(
        loaded,
        max_applications_per_day=daily_limit,
        auto_apply=replace(
            loaded.auto_apply,
            min_batch_size=12,
            max_batch_size=15,
            min_interval_hours=1,
            max_interval_hours=2,
            start_hour=8,
            end_hour=22,
            timezone="Europe/Moscow",
        ),
    )


def test_planned_batch_size_targets_daily_quota_at_morning_start(
    tmp_path: Path,
) -> None:
    app_settings = _quota_settings(tmp_path)
    database = Database(app_settings.database_path)
    database.init()

    batch = planned_batch_size(
        app_settings, database, datetime(2026, 11, 8, 5, 0, tzinfo=UTC)
    )

    assert batch == 12


def test_planned_batch_size_uses_max_batch_late_in_the_window(
    tmp_path: Path,
) -> None:
    app_settings = _quota_settings(tmp_path)
    database = Database(app_settings.database_path)
    database.init()

    batch = planned_batch_size(
        app_settings, database, datetime(2026, 11, 8, 18, 0, tzinfo=UTC)
    )

    assert batch == 15


def test_planned_batch_size_never_exceeds_daily_limit(tmp_path: Path) -> None:
    app_settings = _quota_settings(tmp_path, daily_limit=10)
    database = Database(app_settings.database_path)
    database.init()

    batch = planned_batch_size(
        app_settings, database, datetime(2026, 11, 8, 6, 0, tzinfo=UTC)
    )

    assert batch == 10


def test_questionnaire_blocked_vacancy_stays_in_the_queue(tmp_path: Path) -> None:
    app_settings = auto_mode_settings(tmp_path)
    database = Database(app_settings.database_path)
    database.init()
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY,
        PageState.VACANCY_LOADED,
        "Example",
        "Build Python services",
    )

    class QuestionnaireService:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def auto_apply(self, job_id: str):
            self.calls.append(job_id)
            database.transition(
                job_id,
                VacancyStatus.PENDING_APPROVAL,
                VacancyStatus.APPLY_FAILED,
                error_text="questionnaire_unanswered:нет гражданства",
            )
            return ApprovalResult(False, "Application failed")

    service = QuestionnaireService()

    result = asyncio.run(
        process_vacancy(
            SUMMARY,
            app_settings,
            database,
            FakeHHClient(details),
            FakeAnalyzer(),
            telegram,
            approval_service=service,
            now_factory=lambda: NOW,
        )
    )

    vacancy = database.get(SUMMARY.id)
    assert result.outcome == "telegram_card"
    assert vacancy is not None
    assert vacancy.status is VacancyStatus.PENDING_APPROVAL
    assert SUMMARY.id in [item.id for item in database.pending()]
    assert telegram.previews == [(SUMMARY.id, True)]
