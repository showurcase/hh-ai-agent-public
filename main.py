from __future__ import annotations
import argparse
import asyncio
import hashlib
import logging
import math
import random
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import html
from aiogram.exceptions import TelegramAPIError
from aiogram.types import FSInputFile
from mascots import get_pesec_image, is_pesec_enabled

from ai_analyzer import AnalysisError, VacancyAnalyzer
from approval import ApprovalGuard, ApprovalService
from browser_backend import BrowserLaunchError, create_browser_backend
from config import ConfigError, Settings, load_settings
from database import (
    APPLY_MAX_ATTEMPTS,
    TERMINAL_APPLY_ERRORS,
    Database,
    SearchRun,
    VacancyStatus,
)
from fit_summary import normalize_fit_summary
from hh_client import HHClient, PageState, VacancySummary
from instance_lock import AlreadyRunningError, single_instance
from llm.base import LLMProvider
from llm.errors import LLMError
from llm.factory import create_llm_provider
from llm.mistral_keys import MistralKeyCipher, MistralKeyManager
from llm.types import LLMRequest
from logging_setup import configure_logging
from queue_page import QueuePageServer
from tg_bot import AgentControl, TelegramService
from vacancy_filter import vacancy_rejection_reason
from version import __version__
from mascots import (
    enable_pesec_mode,
    pesec_on_analyzing,
    pesec_on_applied,
    pesec_on_approve,
    pesec_on_reject,
    show_pesec_startup,
)

logger = logging.getLogger(__name__)
ANALYSIS_MAX_ATTEMPTS = 3



@dataclass(frozen=True)
class VacancyProcessResult:
    outcome: str
    reason: str = ""
    page_state: PageState | None = None


@dataclass
class SearchRunStats:
    query_count: int = 0
    found_results: int = 0
    new_vacancies: int = 0
    duplicates: int = 0
    rejected_by_filter: int = 0
    rejected_by_llm: int = 0
    telegram_cards: int = 0
    auto_applications: int = 0
    error_count: int = 0
    rejection_reasons: Counter[str] = field(default_factory=Counter)
    error_reasons: Counter[str] = field(default_factory=Counter)
    last_safe_error: str = ""
    read_attempts: int = 0
    technical_failures: int = 0
    consecutive_page_errors: int = 0

    def record_error(self, reason: str) -> None:
        self.error_count += 1
        self.error_reasons[reason] += 1
        self.last_safe_error = reason

    def record(self, result: VacancyProcessResult) -> None:
        if result.outcome == "rejected_by_filter":
            self.rejected_by_filter += 1
            self.rejection_reasons[result.reason or "filter_rejected"] += 1
        elif result.outcome == "rejected_by_llm":
            self.rejected_by_llm += 1
            self.rejection_reasons["llm_rejected"] += 1
        elif result.outcome == "telegram_card":
            self.telegram_cards += 1
        elif result.outcome == "auto_applied":
            self.auto_applications += 1
        elif result.outcome == "other_error":
            self.record_error(result.reason or "other_error")

        if result.page_state is None:
            return
        self.read_attempts += 1
        if result.page_state is PageState.PAGE_STRUCTURE_CHANGED:
            self.consecutive_page_errors += 1
        else:
            self.consecutive_page_errors = 0
        if result.page_state in {
            PageState.ACCESS_DENIED,
            PageState.CAPTCHA_DETECTED,
            PageState.NETWORK_ERROR,
        }:
            self.technical_failures += 1

    def circuit_reason(self, settings: Settings) -> str:
        if self.consecutive_page_errors >= settings.circuit_breaker_page_errors:
            return "page_structure_changed"
        if (
            self.read_attempts >= settings.circuit_breaker_min_sample
            and self.technical_failures / self.read_attempts
            >= settings.circuit_breaker_unknown_ratio
        ):
            return "technical_failure_ratio"
        return ""


def next_auto_batch_at(
    after: datetime, settings: Settings, *, interval_hours: int = 0
) -> datetime:
    schedule = settings.auto_apply
    local = after.astimezone(ZoneInfo(schedule.timezone))
    window_start = local.replace(
        hour=schedule.start_hour, minute=0, second=0, microsecond=0
    )
    window_end = local.replace(
        hour=schedule.end_hour, minute=0, second=0, microsecond=0
    )
    if local < window_start:
        return window_start
    if local >= window_end:
        return window_start + timedelta(days=1)
    candidate = local + timedelta(hours=interval_hours)
    return candidate if candidate < window_end else window_start + timedelta(days=1)


def planned_batch_size(
    settings: Settings, database: Database, now: datetime
) -> int:
    """How many applications this batch needs to reach the daily target.

    Uses the remaining daily slots and the remaining batches in the working
    window, so the bot does not stop at 5-6 applications when the user wants
    ~100 per day. Never exceeds the configured max batch or the daily limit.
    """
    remaining = database.available_application_slots(
        daily_limit=settings.max_applications_per_day, now=now
    )
    if remaining <= 0:
        return 0
    schedule = settings.auto_apply
    local = now.astimezone(ZoneInfo(schedule.timezone))
    window_end = local.replace(
        hour=schedule.end_hour, minute=0, second=0, microsecond=0
    )
    minutes_left = max(0.0, (window_end - local).total_seconds() / 60)
    if minutes_left <= 0:
        return 0
    interval_minutes = max(1, schedule.min_interval_hours * 60)
    batches_left = max(1, int(minutes_left // interval_minutes))
    needed = math.ceil(remaining / batches_left)
    return min(
        schedule.max_batch_size,
        max(schedule.min_batch_size, needed),
        remaining,
    )


def forced_batch_size(
    settings: Settings, database: Database, now: datetime
) -> int:
    """Batch for a manual run: ignore the daytime window, keep the daily cap."""
    remaining = database.available_application_slots(
        daily_limit=settings.max_applications_per_day, now=now
    )
    if remaining <= 0:
        return 0
    return min(settings.auto_apply.max_batch_size, remaining)


def next_auto_window_start_after(after: datetime, settings: Settings) -> datetime:
    schedule = settings.auto_apply
    local = after.astimezone(ZoneInfo(schedule.timezone))
    window_start = local.replace(
        hour=schedule.start_hour, minute=0, second=0, microsecond=0
    )
    return window_start if local < window_start else window_start + timedelta(days=1)


def restored_auto_batch_at(
    now: datetime,
    settings: Settings,
    database: Database,
    *,
    interval_hours: int,
) -> datetime:
    """Keep a restarted auto-apply process from starting a second batch early."""
    previous = database.latest_search_run()
    if previous is None:
        return next_auto_batch_at(now, settings)
    try:
        finished_at = datetime.fromisoformat(previous.finished_at)
    except ValueError:
        logger.warning("auto_schedule_restore_failed run_id=%s", previous.id)
        return next_auto_batch_at(now, settings)
    candidate = next_auto_batch_at(
        finished_at, settings, interval_hours=interval_hours
    )
    # A bot that was stopped for days must not fire a batch immediately on
    # restart: walk the schedule forward until it lands after `now`.
    while candidate <= now:
        if interval_hours <= 0:
            # No pause: jump to "now", but still land inside the work window.
            return next_auto_batch_at(now, settings, interval_hours=0)
        candidate = next_auto_batch_at(
            candidate + timedelta(seconds=1),
            settings,
            interval_hours=interval_hours,
        )
    return candidate


def format_auto_batch_report(
    run: SearchRun,
    *,
    auto_applied: int,
    next_batch: datetime | None,
    settings: Settings,
) -> str:
    lines = [
        "Итог автоподачи",
        f"Найдено: {run.found_results}; новых: {run.new_vacancies}.",
        f"Отправлено: {auto_applied}; на ручную проверку: {run.telegram_cards}.",
        (
            "Отсечено фильтрами: "
            f"{run.rejected_by_filter + run.rejected_by_llm}; "
            f"безопасно остановлено: {run.error_count}."
        ),
    ]
    if next_batch is None:
        lines.append("Следующая пачка не назначена: автоподача приостановлена.")
    else:
        local_next = next_batch.astimezone(ZoneInfo(settings.auto_apply.timezone))
        lines.append(
            f"Следующая пачка: {local_next.strftime('%d.%m %H:%M')} "
            f"{settings.auto_apply.timezone}."
        )
    return "\n".join(lines)


async def requeue_failed_application(
    database: Database,
    telegram: TelegramService,
    job_id: str,
    message: str,
) -> str:
    """Give a failed application one more automatic try, then park it in the queue."""
    vacancy = database.get(job_id)
    if vacancy is None or vacancy.status is not VacancyStatus.APPLY_FAILED:
        return "auto_apply_failed"
    if vacancy.error_text in TERMINAL_APPLY_ERRORS:
        # A removed vacancy or an existing response is not a failure to repeat.
        database.transition(
            job_id,
            VacancyStatus.APPLY_FAILED,
            VacancyStatus.SKIPPED,
            error_text=vacancy.error_text,
        )
        logger.info(
            "application_retry_skipped job_id=%s reason=%s", job_id, vacancy.error_text
        )
        return "skipped"
    attempts = database.bump_apply_retry(job_id)
    if attempts < APPLY_MAX_ATTEMPTS:
        database.requeue_failed_for_retry(job_id)
        logger.info(
            "application_retry_scheduled job_id=%s attempt=%s message=%s",
            job_id,
            attempts,
            message,
        )
        return "retry_scheduled"
    database.requeue_failed_for_retry(job_id, VacancyStatus.PENDING_APPROVAL)
    logger.info(
        "application_retry_exhausted job_id=%s attempt=%s message=%s",
        job_id,
        attempts,
        message,
    )
    parked = database.get(job_id)
    if parked is not None:
        try:
            await telegram.send_preview(parked, include_actions=True)
        except TelegramAPIError:
            logger.warning("retry_card_failed job_id=%s", job_id)
    return "telegram_card"


async def reprocess_failed_vacancies(
    settings: Settings,
    database: Database,
    hh_client: HHClient,
    analyzer: VacancyAnalyzer,
    telegram: TelegramService,
    approval_service: ApprovalService | None,
    limit: int,
) -> int:
    """Re-read the failures that never produced a cover letter."""
    if limit < 1:
        return 0
    candidates = database.reprocess_candidates(limit)
    if not candidates:
        return 0
    logger.info("vacancy_reprocess_started count=%s", len(candidates))
    reprocessed = 0
    for job_id in candidates:
        vacancy = database.get(job_id)
        if vacancy is None or not database.reset_for_reprocess(job_id):
            continue
        summary = VacancySummary(
            job_id, vacancy.title, vacancy.url, vacancy.search_query
        )
        try:
            await process_vacancy(
                summary,
                settings,
                database,
                hh_client,
                analyzer,
                telegram,
                approval_service=approval_service,
            )
        except Exception as exc:
            logger.warning(
                "vacancy_reprocess_failed job_id=%s error_type=%s error=%s",
                job_id,
                type(exc).__name__,
                exc,
            )
            database.transition(
                job_id,
                VacancyStatus.DISCOVERED,
                VacancyStatus.APPLY_FAILED,
                error_text=f"{type(exc).__name__}: {exc}",
            )
        reprocessed += 1
    return reprocessed


async def retry_failed_applications(
    database: Database,
    approval_service: ApprovalService,
    telegram: TelegramService,
    limit: int,
) -> int:
    """Re-run the applications that failed their first attempt."""
    if limit < 1:
        return 0
    candidates = database.retry_candidates(limit)
    if not candidates:
        return 0
    logger.info("application_retry_started count=%s", len(candidates))
    retried = 0
    for job_id in candidates:
        vacancy = database.get(job_id)
        if vacancy is None:
            continue
        result = await approval_service.auto_apply(job_id)
        retried += 1
        if result.ok:
            try:
                await telegram.notify(
                    f"✓ Повторный автоотклик отправлен: {vacancy.title}\n{vacancy.url}"
                )
            except TelegramAPIError:
                logger.warning("retry_notification_failed job_id=%s", job_id)
            continue
        await requeue_failed_application(database, telegram, job_id, result.message)
    return retried


async def process_vacancy(
    summary: VacancySummary,
    settings: Settings,
    database: Database,
    hh_client: HHClient,
    analyzer: VacancyAnalyzer,
    telegram: TelegramService,
    *,
    approval_service: ApprovalService | None = None,
    now_factory: Callable[[], datetime] | None = None,
) -> VacancyProcessResult:
    pesec_on_analyzing(summary.title)
    clock = now_factory or (lambda: datetime.now(UTC))
    now = clock()

    async def send_failure_screenshot(caption: str) -> None:
        path = getattr(hh_client, "last_failure_screenshot", None)
        if path is None or not hasattr(telegram, "notify_screenshot"):
            return
        try:
            await telegram.notify_screenshot(path, caption)
        except Exception as exc:
            logger.warning(
                "failure_screenshot_notification_failed job_id=%s error=%s",
                summary.id,
                exc,
            )
    existing = database.get(summary.id)
    if summary.previously_sent:
        return VacancyProcessResult("ignored")
    if existing is not None and (
        existing.status is not VacancyStatus.DISCOVERED
        or existing.llm_decision is not None
    ):
        return VacancyProcessResult("ignored")
    details = await hh_client.read_vacancy(
        summary,
        getattr(hh_client, "captcha_solver", None) or telegram.request_captcha,
    )
    description_hash = (
        hashlib.sha256(details.description.encode()).hexdigest()
        if details.description
        else ""
    )
    if existing is None:
        if not database.discover(
            job_id=summary.id,
            title=summary.title,
            company=details.company,
            company_url=details.company_url,
            url=summary.url,
            description_hash=description_hash,
            search_query=summary.search_query,
            discovered_at=now,
        ):
            return VacancyProcessResult("ignored")
        logger.info("vacancy_discovered job_id=%s", summary.id)
    else:
        logger.info("vacancy_resumed job_id=%s", summary.id)
    if details.state is PageState.ALREADY_RESPONDED:
        database.transition(
            summary.id,
            VacancyStatus.DISCOVERED,
            VacancyStatus.SKIPPED,
            error_text="already_responded",
        )
        logger.info("vacancy_skipped job_id=%s reason=already_responded", summary.id)
        return VacancyProcessResult("skipped", "already_responded", details.state)
    if details.state is not PageState.VACANCY_LOADED:
        error = details.error or details.state.value
        if details.state in (
            PageState.VACANCY_REMOVED,
            PageState.FOREIGN_VACANCY,
            PageState.INACCESSIBLE,
        ):
            # Nothing to retry and nothing to show: the vacancy is gone, sits in
            # another country, or HH refuses to show it to this account.
            reason = {
                PageState.FOREIGN_VACANCY: "foreign_vacancy",
                PageState.INACCESSIBLE: "vacancy_inaccessible",
            }.get(details.state, "vacancy_removed")
            database.transition(
                summary.id,
                VacancyStatus.DISCOVERED,
                VacancyStatus.SKIPPED,
                error_text=reason,
            )
            logger.info("vacancy_skipped job_id=%s reason=%s", summary.id, reason)
            return VacancyProcessResult("skipped", reason, details.state)
        database.transition(
            summary.id,
            VacancyStatus.DISCOVERED,
            VacancyStatus.APPLY_FAILED,
            error_text=error,
        )
        if details.state is PageState.CAPTCHA_DETECTED:
            await telegram.notify(f"CAPTCHA was not completed for vacancy {summary.id}.")
        logger.error(
            "vacancy_read_failed job_id=%s state=%s error=%s url=%s",
            summary.id,
            details.state.value,
            error,
            summary.url,
        )
        screenshot = getattr(hh_client, "last_failure_screenshot", None)
        if screenshot is not None:
            await telegram.notify_screenshot(
                screenshot, f"Экран при сбое чтения: {summary.title}\n{summary.url}"
            )
            hh_client.last_failure_screenshot = None
        return VacancyProcessResult("other_error", details.state.value, details.state)

    rejection = vacancy_rejection_reason(
        title=summary.title,
        company=details.company,
        description=details.description,
        excluded_positions=settings.profile.candidate.excluded_positions,
        excluded_companies=settings.profile.candidate.excluded_companies,
        excluded_keywords=settings.profile.candidate.excluded_keywords,
    )
    if rejection:
        database.transition(
            summary.id,
            VacancyStatus.DISCOVERED,
            VacancyStatus.REJECTED_BY_FILTER,
            llm_reason=f"Excluded by profile filter: {rejection}",
        )
        logger.info("vacancy_rejected job_id=%s source=filter", summary.id)
        return VacancyProcessResult(
            "rejected_by_filter", rejection, PageState.VACANCY_LOADED
        )

    try:
        suitability = await analyzer.assess(summary.title, details.description)
    except AnalysisError as exc:
        database.mark_analysis_failed(
            summary.id,
            error_type=exc.error_type,
            now=clock(),
            retry_after=timedelta(minutes=settings.check_interval_minutes),
            max_attempts=ANALYSIS_MAX_ATTEMPTS,
        )
        await telegram.notify_analysis_failed(summary.title, summary.url, exc.error_type)
        logger.warning("vacancy_analysis_failed job_id=%s error_type=%s", summary.id, exc.error_type)
        return VacancyProcessResult(
            "other_error",
            f"analysis_failed:{exc.error_type}",
            PageState.VACANCY_LOADED,
        )

    if not suitability.suitable:
        pesec_on_reject(summary.title, reason=f"LLM: {suitability.reason}")
        database.transition(
            summary.id,
            VacancyStatus.DISCOVERED,
            VacancyStatus.REJECTED_BY_LLM,
            llm_decision=False,
            llm_reason=suitability.reason,
            confidence=suitability.confidence,
            error_text="",
        )
        logger.info("vacancy_rejected job_id=%s source=llm", summary.id)
        return VacancyProcessResult(
            "rejected_by_llm", suitability.reason, PageState.VACANCY_LOADED
        )

    pesec_on_approve(summary.title, score=suitability.confidence)

    fit_summary = normalize_fit_summary(suitability.fit_points)
    company = await hh_client.read_company_details(details.company_url)
    database.store_company_details(
        summary.id, rating=company.rating, reviews_count=company.reviews_count
    )

    letter = await analyzer.generate_cover_letter(
        summary.title, details.description, company_name=details.company
    )
    if not letter.strip():
        database.transition(
            summary.id,
            VacancyStatus.DISCOVERED,
            VacancyStatus.APPLY_FAILED,
            error_text="cover_letter_failed",
        )
        await telegram.notify_analysis_failed(summary.title, summary.url, "cover_letter_failed")
        return VacancyProcessResult(
            "other_error", "cover_letter_failed", PageState.VACANCY_LOADED
        )

    include_actions = settings.app_mode != "dry_run"
    is_ai_priority = bool(suitability.is_ai_priority)
    if settings.app_mode == "dry_run":
        database.store_analysis(
            summary.id,
            cover_letter=letter,
            llm_decision=True,
            llm_reason=suitability.reason,
            confidence=suitability.confidence,
            fit_summary=fit_summary,
            is_ai_priority=is_ai_priority,
        )
    elif not database.request_approval(
        job_id=summary.id,
        cover_letter=letter,
        llm_decision=True,
        llm_reason=suitability.reason,
        confidence=suitability.confidence,
        fit_summary=fit_summary,
        is_ai_priority=is_ai_priority,
        now=now,
    ):
        return VacancyProcessResult(
            "other_error", "approval_transition_failed", PageState.VACANCY_LOADED
        )
    elif (
        not details.foreign_remote
        and (not is_ai_priority or database.manual_review_disabled())
        and (
            settings.app_mode == "auto"
            or (
                settings.auto_apply.enabled
                and suitability.confidence >= settings.auto_apply.min_confidence
            )
        )
    ):
        if approval_service is None:
            return VacancyProcessResult(
                "other_error", "auto_apply_service_unavailable", PageState.VACANCY_LOADED
            )
        auto_result = await approval_service.auto_apply(summary.id)
        vacancy = database.get(summary.id)
        if auto_result.ok:
            pesec_on_applied(summary.title)
            sent_title = vacancy.title if vacancy else summary.title
            sent_url = vacancy.url if vacancy else summary.url
            sent_company = vacancy.company if vacancy else ""
            lines = [f"✓ Автоотклик отправлен: {sent_title}"]
            if sent_company:
                lines.append(sent_company)
            lines.append(sent_url)
            try:
                await telegram.notify("\n".join(lines))
            except TelegramAPIError:
                logger.warning("auto_application_notification_failed job_id=%s", summary.id)
            return VacancyProcessResult("auto_applied", page_state=PageState.VACANCY_LOADED)
        if vacancy is not None and vacancy.status is VacancyStatus.SKIPPED:
            logger.info(
                "auto_application_skipped job_id=%s reason=%s",
                summary.id,
                vacancy.error_text,
            )
            return VacancyProcessResult("skipped", vacancy.error_text, PageState.VACANCY_LOADED)
        if vacancy is not None and "questionnaire" in vacancy.error_text:
            await send_failure_screenshot(
                f"Экран при сбое: {vacancy.title}\n{vacancy.url}"
            )
            # Keep it in the queue: an unanswered form needs the owner, and a
            # card alone disappears from the chat.
            database.transition(
                vacancy.id,
                VacancyStatus.APPLY_FAILED,
                VacancyStatus.PENDING_APPROVAL,
            )
            vacancy = database.get(vacancy.id) or vacancy
            try:
                await telegram.notify_questionnaire_required(
                    vacancy.title, vacancy.url
                )
                await telegram.send_preview(
                    vacancy,
                    include_actions=True,
                    status_badge="📋 <b>ТРЕБУЕТСЯ АНКЕТА / ТЕСТОВОЕ НА САЙТЕ</b>",
                )
            except TelegramAPIError:
                return VacancyProcessResult(
                    "other_error", "telegram_error", PageState.VACANCY_LOADED
                )
            return VacancyProcessResult(
                "telegram_card", "questionnaire_required", PageState.VACANCY_LOADED
            )
        if vacancy is not None and vacancy.status.value == "pending_approval":
            try:
                await telegram.send_preview(vacancy, include_actions=True)
            except TelegramAPIError:
                return VacancyProcessResult(
                    "other_error", "telegram_error", PageState.VACANCY_LOADED
                )
            return VacancyProcessResult("telegram_card", page_state=PageState.VACANCY_LOADED)
        told_title = vacancy.title if vacancy else summary.title
        told_url = vacancy.url if vacancy else summary.url
        outcome = await requeue_failed_application(
            database, telegram, summary.id, auto_result.message
        )
        if outcome == "retry_scheduled":
            try:
                await telegram.notify(
                    f"↻ Отклик не прошёл, повторю в следующем цикле: {told_title}\n"
                    f"{told_url}\n"
                    f"{auto_result.message}"
                )
            except TelegramAPIError:
                logger.warning("auto_application_notification_failed job_id=%s", summary.id)
            return VacancyProcessResult(
                "retry_scheduled", "auto_apply_failed", PageState.VACANCY_LOADED
            )
        await send_failure_screenshot(f"Экран при сбое: {told_title}\n{told_url}")
        try:
            await telegram.notify(
                f"✗ Автоотклик не отправлен: {told_title}\n"
                f"{told_url}\n"
                f"{auto_result.message}"
            )
        except TelegramAPIError:
            logger.warning("auto_application_notification_failed job_id=%s", summary.id)
        return VacancyProcessResult(
            "other_error", "auto_apply_failed", PageState.VACANCY_LOADED
        )
    try:
        await telegram.send_preview(
            database.get(summary.id), include_actions=include_actions
        )
    except TelegramAPIError:
        return VacancyProcessResult(
            "other_error", "telegram_error", PageState.VACANCY_LOADED
        )
    return VacancyProcessResult("telegram_card", page_state=PageState.VACANCY_LOADED)


async def run_search_cycle(
    settings: Settings,
    database: Database,
    hh_client: HHClient,
    analyzer: VacancyAnalyzer,
    telegram: TelegramService,
    control: AgentControl,
    *,
    approval_service: ApprovalService | None = None,
    auto_apply_batch_limit: int | None = None,
    now_factory: Callable[[], datetime] | None = None,
) -> SearchRun:
    if auto_apply_batch_limit is not None and auto_apply_batch_limit < 1:
        raise ValueError("auto_apply_batch_limit must be positive")
    now = now_factory or (lambda: datetime.now(UTC))
    started_at = now()
    stats = SearchRunStats()
    handled_ids: set[str] = set()
    state = "completed"
    circuit_reason = ""
    failure: Exception | None = None
    logger.info(
        "search_cycle_started queries=%s", len(settings.profile.hh.search_queries)
    )

    if approval_service is not None and settings.app_mode != "dry_run":
        # One failed vacancy per cycle max: reprocessing reloads the page and
        # re-runs LLM analysis, so a larger retry batch delays fresh searches.
        retry_limit = 1
        try:
            await reprocess_failed_vacancies(
                settings,
                database,
                hh_client,
                analyzer,
                telegram,
                approval_service,
                retry_limit,
            )
            await retry_failed_applications(
                database,
                approval_service,
                telegram,
                retry_limit,
            )
        except Exception as exc:
            logger.warning(
                "application_retry_failed error_type=%s error=%s",
                type(exc).__name__,
                exc,
            )

    uncertain_deliveries = 0

    async def process(summary: VacancySummary) -> None:
        nonlocal circuit_reason, state, uncertain_deliveries
        if summary.id in handled_ids:
            return
        handled_ids.add(summary.id)
        result = await process_vacancy(
            summary,
            settings,
            database,
            hh_client,
            analyzer,
            telegram,
            approval_service=approval_service,
            now_factory=now,
        )
        stats.record(result)
        if result.outcome == "auto_applied":
            uncertain_deliveries = 0
        if result.reason == "auto_apply_failed":
            failed_application = database.get(summary.id)
            if (
                failed_application is not None
                and failed_application.submit_attempted_at is not None
            ):
                # One unconfirmed submission is an ordinary exception: the
                # vacancy is never retried, so it cannot double-apply. Only a
                # streak means HH itself is misbehaving.
                uncertain_deliveries += 1
                logger.warning(
                    "application_delivery_uncertain job_id=%s streak=%s",
                    summary.id,
                    uncertain_deliveries,
                )
                if uncertain_deliveries >= 2:
                    circuit_reason = "application_delivery_uncertain"
        if not circuit_reason:
            circuit_reason = stats.circuit_reason(settings)
        if circuit_reason:
            state = "paused_by_circuit_breaker"
            control.paused = True
            control.circuit_reason = circuit_reason

    try:
        database.expire_approved(now())
        database.requeue_due_analysis_failures(
            now(), max_attempts=ANALYSIS_MAX_ATTEMPTS
        )
        for vacancy in database.unprocessed_discovered():
            if control.paused or (
                auto_apply_batch_limit is not None
                and stats.auto_applications >= auto_apply_batch_limit
            ):
                break
            await process(
                VacancySummary(
                    vacancy.id,
                    vacancy.title,
                    vacancy.url,
                    vacancy.search_query,
                )
            )
        for query in settings.profile.hh.search_queries:
            if control.paused or (
                auto_apply_batch_limit is not None
                and stats.auto_applications >= auto_apply_batch_limit
            ):
                break
            logger.info("search_query_started query=%r", query)
            search = await hh_client.search_vacancies(
                query,
                settings.profile.hh.areas,
                settings.profile.hh.experience_filters,
                remote_only=settings.profile.hh.remote_only,
            )
            stats.query_count += 1
            stats.found_results += search.found_results
            stats.new_vacancies += sum(
                not summary.previously_sent for summary in search.summaries
            )
            stats.duplicates += search.duplicates
            logger.info(
                "search_query_finished query=%r found=%s new=%s duplicates=%s",
                query,
                search.found_results,
                sum(not summary.previously_sent for summary in search.summaries),
                search.duplicates,
            )
            if search.error_reason:
                stats.record_error(search.error_reason)
                control.consecutive_search_errors += 1
            else:
                control.consecutive_search_errors = 0
            if (
                control.consecutive_search_errors
                >= settings.circuit_breaker_page_errors
            ):
                circuit_reason = "search_errors"
                state = "paused_by_circuit_breaker"
                control.paused = True
                control.circuit_reason = circuit_reason
            for summary in search.summaries:
                if control.paused or (
                    auto_apply_batch_limit is not None
                    and stats.auto_applications >= auto_apply_batch_limit
                ):
                    break
                await process(summary)
        if not control.paused:
            await hh_client.check_messages(telegram.notify)
    except Exception as exc:
        state = "failed"
        stats.record_error(type(exc).__name__)
        failure = exc

    if circuit_reason:
        notification = (
            f"Поиск приостановлен: {circuit_reason}. "
            "Проверьте /diagnostics и выполните /resume после устранения причины."
        )
    elif (
        auto_apply_batch_limit is None
        and state == "completed"
        and stats.telegram_cards == 0
    ):
        reasons = (stats.rejection_reasons + stats.error_reasons).most_common(3)
        reason_text = ", ".join(f"{name}={count}" for name, count in reasons)
        notification = (
            f"Цикл завершён: найдено {stats.found_results}, "
            f"новых {stats.new_vacancies}, карточек: 0."
            + (f" Причины: {reason_text}." if reason_text else "")
        )
    else:
        notification = ""

    run_id = database.save_search_run(
        started_at=started_at,
        finished_at=now(),
        state=state,
        query_count=stats.query_count,
        found_results=stats.found_results,
        new_vacancies=stats.new_vacancies,
        duplicates=stats.duplicates,
        rejected_by_filter=stats.rejected_by_filter,
        rejected_by_llm=stats.rejected_by_llm,
        telegram_cards=stats.telegram_cards,
        error_count=stats.error_count,
        rejection_reasons=dict(stats.rejection_reasons),
        error_reasons=dict(stats.error_reasons),
        last_safe_error=stats.last_safe_error,
        circuit_reason=circuit_reason,
    )
    if failure is not None:
        raise failure
    if notification:
        try:
            await telegram.notify(notification)
        except TelegramAPIError:
            database.record_search_run_error(run_id, "telegram_error")
    result = database.latest_search_run()
    if result is None:
        raise RuntimeError("search run was not saved")
    logger.info(
        "search_cycle_finished state=%s cards=%s auto_applied=%s errors=%s",
        result.state,
        result.telegram_cards,
        stats.auto_applications,
        result.error_count,
    )
    return result


async def retry_due_analyses(
    settings: Settings,
    database: Database,
    hh_client: HHClient,
    analyzer: VacancyAnalyzer,
    telegram: TelegramService,
    control: AgentControl,
    *,
    now_factory: Callable[[], datetime] | None = None,
) -> None:
    clock = now_factory or (lambda: datetime.now(UTC))
    now = clock()
    # ponytail: one worker owns this lifecycle; add per-row claims if
    # multi-worker deployments matter.
    database.requeue_due_analysis_failures(
        now, max_attempts=ANALYSIS_MAX_ATTEMPTS
    )
    for vacancy in database.unprocessed_discovered():
        if control.paused:
            break
        await process_vacancy(
            VacancySummary(
                vacancy.id,
                vacancy.title,
                vacancy.url,
                vacancy.search_query,
            ),
            settings,
            database,
            hh_client,
            analyzer,
            telegram,
            now_factory=clock,
        )


async def _session_alive(hh_client: HHClient, settings: Settings) -> bool:
    """True when the hh.ru account is still logged in."""
    try:
        return bool(await hh_client.ensure_login())
    except Exception as exc:
        logger.warning(
            "login_check_failed error_type=%s error=%s", type(exc).__name__, exc
        )
        return True


async def agent_loop(
    settings: Settings,
    database: Database,
    hh_client: HHClient,
    analyzer: VacancyAnalyzer,
    telegram: TelegramService,
    control: AgentControl,
    approval_service: ApprovalService | None = None,
    *,
    now_factory: Callable[[], datetime] | None = None,
    randint: Callable[[int, int], int] = random.randint,
) -> None:
    clock = now_factory or (lambda: datetime.now(UTC))
    next_auto_batch: datetime | None = None
    while True:
        wake_timeout_seconds = settings.check_interval_minutes * 60
        if next_auto_batch is not None and not control.paused:
            until_due = (next_auto_batch - clock()).total_seconds()
            wake_timeout_seconds = max(1.0, min(wake_timeout_seconds, until_due))
        if control.paused:
            control.next_run_at = None
        elif settings.auto_apply.enabled:
            now = clock()
            next_auto_batch = next_auto_batch or restored_auto_batch_at(
                now,
                settings,
                database,
                interval_hours=randint(
                    settings.auto_apply.min_interval_hours,
                    settings.auto_apply.max_interval_hours,
                ),
            )
            forced = control.force_run
            if forced:
                control.force_run = False
            if forced or now >= next_auto_batch:
                batch_size = (
                    forced_batch_size(settings, database, clock())
                    if forced
                    else planned_batch_size(settings, database, now)
                )
                if not batch_size and forced:
                    logger.info(
                        "manual_run_skipped reason=daily_limit applied_today=%s limit=%s",
                        database.applied_today(clock()),
                        settings.max_applications_per_day,
                    )
                    try:
                        await telegram.notify(
                            "Дневной лимит откликов исчерпан "
                            f"({database.applied_today(clock())}"
                            f" / {settings.max_applications_per_day}). "
                            "Поиск пропущен."
                        )
                    except Exception as exc:
                        logger.warning("manual_run_notify_failed error=%s", exc)
                if batch_size and not await _session_alive(hh_client, settings):
                    # An expired hh.ru session makes every application fail with
                    # "final application button was not found"; stop instead of
                    # burning the whole day on anonymous pages.
                    control.paused = True
                    control.next_run_at = None
                    logger.error("hh_login_required cycle_skipped")
                    try:
                        await telegram.notify(
                            "⚠️ Бот разлогинен на hh.ru: отклики не отправляются.\n"
                            "Нужен повторный вход в аккаунт."
                        )
                    except Exception as exc:
                        logger.warning("login_notification_failed error=%s", exc)
                    batch_size = 0
                if not batch_size and not forced:
                    # Nothing to send (daily cap or empty window): do not spin
                    # every second with a zero interval.
                    next_auto_batch = clock() + timedelta(minutes=5)
                if batch_size:
                    applications_before = database.applied_today(now)
                    run = await run_search_cycle(
                        settings,
                        database,
                        hh_client,
                        analyzer,
                        telegram,
                        control,
                        approval_service=approval_service,
                        auto_apply_batch_limit=batch_size,
                        now_factory=clock,
                    )
                    if control.paused:
                        control.next_run_at = None
                    else:
                        next_auto_batch = next_auto_batch_at(
                            clock(),
                            settings,
                            interval_hours=randint(
                                settings.auto_apply.min_interval_hours,
                                settings.auto_apply.max_interval_hours,
                            ),
                        )
                        control.next_run_at = next_auto_batch
                    try:
                        await telegram.notify(
                            format_auto_batch_report(
                                run,
                                auto_applied=max(
                                    0,
                                    database.applied_today(clock())
                                    - applications_before,
                                ),
                                next_batch=control.next_run_at,
                                settings=settings,
                            )
                        )
                    except TelegramAPIError:
                        logger.warning("auto_batch_report_notification_failed")
                else:
                    next_auto_batch = next_auto_window_start_after(now, settings)
                    control.next_run_at = next_auto_batch
            else:
                control.next_run_at = next_auto_batch
            if control.next_run_at is not None:
                wake_timeout_seconds = max(
                    0.0, (control.next_run_at - clock()).total_seconds()
                )
        else:
            run = await run_search_cycle(
                settings, database, hh_client, analyzer, telegram, control
            )
            control.next_run_at = (
                None
                if control.paused
                else datetime.fromisoformat(run.finished_at)
                + timedelta(minutes=settings.check_interval_minutes)
            )
            wake_timeout_seconds = settings.check_interval_minutes * 60
        wake_received = False
        try:
            await asyncio.wait_for(
                control.wake_event.wait(), wake_timeout_seconds
            )
            wake_received = True
        except TimeoutError:
            pass
        finally:
            control.wake_event.clear()
            control.next_run_at = None
        if wake_received:
            next_auto_batch = None


def _ensure_mistral_key_loaded(settings: Settings, database: Database) -> None:
    if settings.llm.provider == "mistral" and settings.llm.mistral_api_key:
        if not database.mistral_encrypted_keys():
            try:
                cipher = MistralKeyCipher(settings.llm.mistral_keys_master_key)
                protected = cipher.protect(settings.llm.mistral_api_key)
                database.add_mistral_key(
                    encrypted_key=protected.encrypted_key,
                    key_hmac=protected.key_hmac,
                    suffix=protected.suffix,
                    now=datetime.now(UTC),
                )
            except Exception as exc:
                logger.warning("mistral_key_bootstrap_failed error=%s", exc)



async def resume_bumper_loop(
    settings: Settings,
    database: Database,
    hh_client: HHClient,
    telegram: TelegramService,
    control: AgentControl,
    *,
    now_factory: Callable[[], datetime] | None = None,
) -> None:
    clock = now_factory or (lambda: datetime.now(UTC))
    msk_tz = ZoneInfo("Europe/Moscow")

    logger.info("resume_bumper_loop_started (active 08:00–24:00 MSK)")
    await asyncio.sleep(15)

    while True:
        try:
            utc_now = clock()
            msk_now = utc_now.astimezone(msk_tz)

            # Check window: 08:00 to 24:00 MSK (only daytime MSK)
            if not (8 <= msk_now.hour < 24):
                logger.debug("resume_bump_night_window_skip msk_hour=%s", msk_now.hour)
                await asyncio.sleep(900)
                continue

            last_bump = database.last_successful_resume_bump()
            min_interval = 14400 + random.randint(180, 600)
            if last_bump is not None:
                elapsed = (utc_now - last_bump).total_seconds()
                if elapsed < min_interval:
                    sleep_time = min(900, max(60, int(min_interval - elapsed)))
                    await asyncio.sleep(sleep_time)
                    continue

            logger.info("resume_bumper_triggering_bump msk_time=%s", msk_now.strftime("%H:%M:%S"))
            result = await hh_client.bump_resume()
            database.record_resume_bump(
                utc_now,
                settings.profile.hh.resume_name,
                result.ok,
                result.message,
            )

            if result.ok:
                res_name = html.escape(settings.profile.hh.resume_name)
                time_str = msk_now.strftime("%H:%M")
                next_time_str = (msk_now + timedelta(hours=4, minutes=5)).strftime("%H:%M")
                text = (
                    f"🚀 <b>Резюме поднято в поиске hh.ru!</b>\n\n"
                    f"📄 Резюме: <b>{res_name}</b>\n"
                    f"🕒 Время (МСК): {time_str}\n"
                    f"⏳ Следующее поднятие: после {next_time_str} МСК"
                )
                try:
                    await telegram.notify(text)
                except Exception as exc:
                    logger.debug("resume_bump_notify_failed error=%s", exc)
                if result.retry_at is not None:
                    wait_seconds = (
                        result.retry_at - clock()
                    ).total_seconds() + random.randint(60, 300)
                    await asyncio.sleep(max(60, int(wait_seconds)))
                else:
                    await asyncio.sleep(14400)
            else:
                logger.info("resume_bump_not_performed reason=%s", result.message)
                if result.retry_at is not None:
                    # Sleep until HH says the bump is available again
                    # (+ jitter), instead of hammering the page every 15 min.
                    wait_seconds = (
                        result.retry_at - clock()
                    ).total_seconds() + random.randint(60, 300)
                    sleep_time = min(15000, max(60, int(wait_seconds)))
                    logger.info(
                        "resume_bumper_waiting_until retry_at=%s sleep=%ss",
                        result.retry_at.isoformat(),
                        sleep_time,
                    )
                else:
                    sleep_time = 900
                await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning("resume_bumper_loop_error error=%s", exc)
            await asyncio.sleep(300)

async def run(settings: Settings) -> None:
    database = Database(settings.database_path)
    database.init()
    _ensure_mistral_key_loaded(settings, database)
    queue_page = QueuePageServer(settings, database)
    try:
        await queue_page.start()
    except OSError as exc:
        logger.warning("queue_page_start_failed error=%s", exc)
    backend = create_browser_backend(settings)
    llm_provider = create_llm_provider(settings, database)
    mistral_keys = (
        llm_provider if isinstance(llm_provider, MistralKeyManager) else None
    )
    telegram: TelegramService | None = None
    try:
        context = await backend.start()
        guard = ApprovalGuard(settings, database)
        hh_client = HHClient(context, settings, database, guard)
        if not await hh_client.ensure_login():
            raise RuntimeError(
                "HH.ru login is required. Run with BROWSER_HEADLESS=false and sign in manually."
            )
        control = AgentControl()
        approval_service = ApprovalService(settings, database, hh_client)
        telegram = TelegramService(
            settings,
            database,
            approval_service,
            control,
            mistral_keys=mistral_keys,
            hh_client=hh_client,
            queue_page=queue_page,
        )
        analyzer = VacancyAnalyzer(settings, llm_provider)
        hh_client.analyzer = analyzer

        async def solve_captcha(
            screenshot: Path, title: str, timeout_seconds: int
        ) -> str | None:
            answer = await analyzer.solve_captcha(screenshot)
            if answer:
                logger.info("captcha_solved_automatically title=%s", title)
                return answer
            logger.info("captcha_manual_fallback title=%s", title)
            return await telegram.request_captcha(screenshot, title, timeout_seconds)

        hh_client.captcha_solver = solve_captcha
        if mistral_keys is not None:
            mistral_keys.set_notifier(telegram.notify)
        if hasattr(telegram, "check_updates"):
            asyncio.create_task(telegram.check_updates(notify=True))
        await asyncio.gather(
            telegram.start_polling(),
            agent_loop(
                settings,
                database,
                hh_client,
                analyzer,
                telegram,
                control,
                approval_service,
            ),
            resume_bumper_loop(
                settings,
                database,
                hh_client,
                telegram,
                control,
            ),
        )
    finally:
        try:
            await queue_page.stop()
        finally:
            try:
                if telegram is not None:
                    await telegram.stop()
            finally:
                try:
                    await llm_provider.close()
                finally:
                    await backend.close()


async def check_llm(
    settings: Settings,
    provider_factory: Callable[[Settings, Database], LLMProvider] = create_llm_provider,
) -> None:
    database = Database(settings.database_path)
    database.init()
    _ensure_mistral_key_loaded(settings, database)
    provider = provider_factory(settings, database)
    try:
        response = await provider.generate_text(
            LLMRequest(
                system_instructions="Return only the word OK.",
                user_content="Reply OK.",
                model=settings.llm.model,
                temperature=0,
                max_output_tokens=min(settings.llm.max_output_tokens, 8),
                timeout_seconds=settings.llm.timeout_seconds,
                operation="healthcheck",
            )
        )
        print(
            f"LLM check: provider={response.provider} model={response.model} "
            f"latency_ms={response.latency_ms} success=true"
        )
    finally:
        await provider.close()


def cli(
    argv: list[str] | None = None,
    *,
    provider_factory: Callable[[Settings, Database], LLMProvider] = create_llm_provider,
) -> int:
    parser = argparse.ArgumentParser(description="Safe personal HH assistant")
    parser.add_argument(
        "--version",
        action="version",
        version=f"HH Agent v{__version__}",
    )
    # Скрытый флаг пасхалки
    parser.add_argument(
        "--pesec",
        "--fox",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--check-llm", action="store_true")

    args = parser.parse_args(argv)

    if args.pesec:
        enable_pesec_mode()
        show_pesec_startup()
    
    try:
        settings = load_settings(args.env_file, args.profile)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
        
    

    if args.check_config:
        print(
            f"Configuration valid: mode={settings.app_mode}, "
            f"browser={settings.browser_backend}"
        )
        return 0
    if args.check_llm:
        try:
            asyncio.run(check_llm(settings, provider_factory))
        except LLMError as exc:
            print(
                f"LLM check failed: provider={settings.llm.provider} "
                f"error_type={exc.category}",
                file=sys.stderr,
            )
            return 1
        return 0
    configure_logging(settings.log_path)
    lock_path = settings.database_path.with_name(f"{settings.database_path.name}.lock")
    try:
        with single_instance(lock_path):
            asyncio.run(run(settings))
    except AlreadyRunningError as exc:
        logger.info("startup_skipped reason=already_running")
        print(exc, file=sys.stderr)
        return 0
    except (BrowserLaunchError, RuntimeError) as exc:
        logger.error("startup_failed error=%s", exc)
        print(exc, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        logger.info("agent_stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())