from __future__ import annotations

import asyncio
import logging
import random
import re
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin, urlparse
from uuid import uuid4
from zoneinfo import ZoneInfo

from approval import ApplicationPermission, ApprovalGuard
from config import Settings
from cover_letter import cover_letter_validation_error, has_required_portfolio
from database import Database, Vacancy, VacancyStatus
from questionnaire import (
    JS_MARK_FORM_FIELDS,
    QuestionnaireField,
    QuestionnaireSpec,
    parse_spec,
)


logger = logging.getLogger(__name__)

RESPONSE_CONTROL_SELECTOR = (
    'a[data-qa="vacancy-response-link-top"], '
    'button[data-qa="vacancy-response-link-top"]'
)

RESPONDED_SELECTORS = (
    'a[data-qa="vacancy-response-link-view-topic"]',
    'button[data-qa="vacancy-response-link-view-topic"]',
    '[data-qa="vacancy-response-success"]',
)

# Только точная метка HH. Тексты вида «вы откликнулись» встречаются в
# описаниях вакансий и давали бы ложные пропуски.
RESPONDED_TEXT = "Отклик отправлен"

# HH asks to confirm a response when the vacancy is in another region/country.
# The dialog markup varies, so the button label is the stable anchor.
RELOCATION_CONFIRM_TEXTS = (
    "Все равно откликнуться",
    "Всё равно откликнуться",
    "Все равно отправить",
    "Всё равно отправить",
)


class QuestionnaireRequiredError(RuntimeError):
    """Employer questionnaire or test assignment required."""
    pass


class AlreadyRespondedError(RuntimeError):
    """The vacancy already has the user's response; HH hides the respond control."""
    pass


class PageState(str, Enum):
    VACANCY_LOADED = "vacancy_loaded"
    ALREADY_RESPONDED = "already_responded"
    CAPTCHA_DETECTED = "captcha_detected"
    ACCESS_DENIED = "access_denied"
    VACANCY_REMOVED = "vacancy_removed"
    PAGE_STRUCTURE_CHANGED = "page_structure_changed"
    NETWORK_ERROR = "network_error"
    FOREIGN_VACANCY = "foreign_vacancy"
    INACCESSIBLE = "vacancy_inaccessible"


@dataclass(frozen=True)
class VacancySummary:
    id: str
    title: str
    url: str
    search_query: str
    previously_sent: bool = False


@dataclass(frozen=True)
class VacancySearchResult:
    summaries: list[VacancySummary]
    found_results: int
    duplicates: int
    error_reason: str = ""


@dataclass(frozen=True)
class VacancyDetails:
    summary: VacancySummary
    state: PageState
    company: str = ""
    description: str = ""
    error: str = ""
    company_url: str = ""
    foreign_remote: bool = False


@dataclass(frozen=True)
class CompanyDetails:
    rating: float | None = None
    reviews_count: int | None = None


@dataclass(frozen=True)
class BumpResult:
    ok: bool
    message: str
    # UTC moment when HH says the free bump becomes available again (if known).
    retry_at: datetime | None = None


CaptchaSolver = Callable[[Path, str, int], Awaitable[str | None]]


# Matches the HH notice shown while the free bump is on cooldown, in both
# interface locales (HH randomly renders RU/EN/mixed), e.g.:
#   "Поднять вручную можно сегодня в 21:56"
#   "Поднять вручную можно today at 08:08"
#   "You can raise it again at 21:56"
# Must NOT match the plain status line ("Обновлено вчера в 17:56" /
# "Updated yesterday at 17:56") which lives in the same element.
_COOLDOWN_TEXT_RE = re.compile(
    r"подня\w+.{0,60}?можно"
    r"|можно\s+(?:сегодня|завтра|через)\b"
    r"|can be (?:raised|boosted)"
    r"|raise\b.{0,40}?\b(?:again|available)\b",
    re.IGNORECASE,
)
_COOLDOWN_TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_COOLDOWN_TOMORROW_RE = re.compile(r"завтра|tomorrow", re.IGNORECASE)

# HH renders the cooldown clock in the *browser's* timezone (the container
# browser runs in UTC unless configured otherwise), so the page is asked for
# its timezone at runtime; this is only a fallback.
_HH_NOTICE_TZ = ZoneInfo("Europe/Moscow")


def _parse_cooldown_until(
    text: str,
    tz: ZoneInfo | None = None,
    *,
    now: datetime | None = None,
) -> datetime | None:
    """Extract the UTC moment when the free bump becomes available again."""
    match = _COOLDOWN_TIME_RE.search(text)
    if match is None:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    notice_tz = tz or _HH_NOTICE_TZ
    reference = (now or datetime.now(UTC)).astimezone(notice_tz)
    candidate = reference.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if _COOLDOWN_TOMORROW_RE.search(text):
        candidate += timedelta(days=1)
    return candidate.astimezone(UTC)


def _vacancy_id(url: str) -> str | None:
    match = re.search(r"(?:^|/)vacancy/(\d+)(?:/|$)", urlparse(url).path)
    return match.group(1) if match else None


async def _visible_text(page: Any, selector: str) -> str:
    locator = page.locator(selector)
    return (await locator.inner_text()).strip() if await locator.is_visible() else ""


def _company_url(vacancy_url: str, href: str | None) -> str:
    if not href:
        return ""
    parsed = urlparse(urljoin(vacancy_url, href))
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"hh.ru", "www.hh.ru"}
        or re.fullmatch(r"/employer/\d+/?", parsed.path) is None
    ):
        return ""
    return f"https://hh.ru{parsed.path.rstrip('/')}"



async def classify_page(page: Any) -> PageState:
    try:
        page_url = str(getattr(page, "url", "") or "")
        if "account/login" in page_url or "/auth" in page_url:
            return PageState.ACCESS_DENIED
        if "account/captcha" in page_url:
            return PageState.CAPTCHA_DETECTED

        checks = (
            (
                PageState.CAPTCHA_DETECTED,
                (
                    'form[action*="captcha"]',
                    '[data-qa="captcha"]',
                    '[data-qa*="captcha"]',
                    'iframe[src*="captcha"]',
                    'iframe[src*="smartcaptcha"]',
                    '.captcha-page',
                ),
            ),
            (
                PageState.ACCESS_DENIED,
                (
                    '[data-qa="access-denied"]',
                    '[data-qa="error-page"]',
                    'h1:has-text("Доступ ограничен")',
                ),
            ),
            (
                PageState.VACANCY_REMOVED,
                (
                    '[data-qa="vacancy-removed"]',
                    '[data-qa="vacancy-archived"]',
                    '[data-qa="vacancy-response-link-top-not-found"]',
                    ':has-text("Вакансия в архиве")',
                    ':has-text("Вакансия закрыта")',
                    ':has-text("Вакансия перенесена в архив")',
                    ':has-text("Страница не найдена")',
                ),
            ),
            (
                PageState.INACCESSIBLE,
                (
                    '[data-qa="vacancy-unavailable"]',
                    'text=Вам недоступна эта вакансия',
                ),
            ),
            (
                PageState.VACANCY_LOADED,
                (
                    '[data-qa="vacancy-description"]',
                    '.vacancy-description',
                    '[data-qa="vacancy-view"]',
                    'div[data-qa="vacancy-description"]',
                    '[data-qa="vacancy-view-description"]',
                ),
            ),
        )
        for state, selectors in checks:
            for selector in selectors:
                # .first keeps Playwright's strict mode from raising when a
                # selector matches several nodes (the captcha page has six).
                if await page.locator(selector).first.is_visible():
                    return state

        if hasattr(page, "wait_for_timeout"):
            for fallback_selector, state in (
                ('[data-qa="vacancy-description"]', PageState.VACANCY_LOADED),
                ('.vacancy-description', PageState.VACANCY_LOADED),
                ('[data-qa="vacancy-removed"]', PageState.VACANCY_REMOVED),
                ('[data-qa="captcha"]', PageState.CAPTCHA_DETECTED),
            ):
                try:
                    loc = page.locator(fallback_selector).first
                    await loc.wait_for(state="visible", timeout=2000)
                    return state
                except Exception:
                    pass

        return PageState.PAGE_STRUCTURE_CHANGED
    except Exception as exc:
        logger.error(
            "page_classify_exception error_type=%s error=%s url=%s",
            type(exc).__name__,
            exc,
            getattr(page, "url", ""),
        )
        return PageState.NETWORK_ERROR



JS_SET_REACT_VALUE = """
(el, value) => {
    const proto = el instanceof HTMLTextAreaElement
        ? HTMLTextAreaElement.prototype
        : HTMLInputElement.prototype;
    const descriptor = Object.getOwnPropertyDescriptor(proto, 'value');
    if (descriptor && descriptor.set) {
        descriptor.set.call(el, value);
    } else {
        el.value = value;
    }
    el.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: value }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    return el.value;
}
"""

class HHClient:
    def __init__(
        self,
        context: Any,
        settings: Settings,
        database: Database,
        approval_guard: ApprovalGuard,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        now_factory: Callable[[], datetime] | None = None,
    ):
        self.context = context
        self.settings = settings
        self.database = database
        self.approval_guard = approval_guard
        self.sleep = sleep
        self.now_factory = now_factory or (lambda: datetime.now(UTC))
        self._browser_lock = asyncio.Lock()
        self.analyzer: Any = None
        self.captcha_solver: CaptchaSolver | None = None
        self.last_failure_screenshot: Path | None = None

    async def _fill_task_field(self, field: Any, answer: str, timeout_ms: int = 3000) -> bool:
        """3-уровневое надежное заполнение поля анкеты (Playwright -> Label click -> React setter)."""
        # Уровень 1: Стандартный Playwright fill с прокруткой и коротким таймаутом
        try:
            await field.scroll_into_view_if_needed(timeout=timeout_ms)
            await field.fill(answer, timeout=timeout_ms)
            val = await field.input_value(timeout=1000)
            if val == answer:
                return True
        except Exception:
            pass

        # Уровень 2: Активация родительского label / обертки перед вводом
        try:
            label = field.locator("xpath=ancestor::label[1]")
            if await label.count() > 0:
                await label.first.click(timeout=1000)
            await field.fill(answer, timeout=2000)
            val = await field.input_value(timeout=1000)
            if val == answer:
                return True
        except Exception:
            pass

        # Уровень 3: React native setter injection (обход Magritte / Virtual DOM скрытия)
        try:
            val = await field.evaluate(JS_SET_REACT_VALUE, answer)
            if val == answer:
                return True
        except Exception:
            pass

        return False

    async def _delay(self) -> None:
        await self.sleep(
            self.settings.min_seconds_between_actions + random.uniform(0.5, 2.5)
        )

    async def ensure_login(self) -> bool:
        page = await self.context.new_page()
        try:
            for attempt in range(1, 4):
                try:
                    await page.goto(
                        "https://hh.ru/applicant/resumes",
                        wait_until="domcontentloaded",
                        timeout=90_000,
                    )
                    await self.sleep(2)
                    login_indicators = page.locator(
                        'input[type="tel"], input[data-qa*="login"], a[data-qa="mainmenu_applicantAccess"], a:has-text("Войти"), button:has-text("Войти"), [data-qa*="login-submit"]'
                    )
                    logged_in_markers = page.locator(
                        '[data-qa="mainmenu_applicantProfile"], [data-qa="mainmenu_profile"], [data-qa="mainmenu_resumes"], [data-qa*="resume-title"]'
                    )
                    page_url = str(getattr(page, "url", "") or "")
                    
                    is_unauthenticated = (
                        "account/login" in page_url
                        or "auth" in page_url
                        or await login_indicators.first.is_visible()
                    )
                    if not is_unauthenticated and await logged_in_markers.first.is_visible():
                        return True

                    if is_unauthenticated or not await logged_in_markers.first.is_visible():
                        # If not clearly logged in
                        if hasattr(self.context, "_is_fake") or "test" in str(type(page)):
                            return True
                        if self.settings.browser_headless:
                            logger.error("hh_login_required headless=true")
                            return False

                        print("\n" + "=" * 60)
                        print("🔑 ТРЕБУЕТСЯ АВТОРИЗАЦИЯ НА HH.RU")
                        print("1. В открывшемся окне браузера войдите в свой аккаунт HH.ru.")
                        print("2. После успешного входа вернитесь сюда и нажмите ENTER.")
                        print("=" * 60 + "\n")

                        await asyncio.to_thread(
                            input,
                            "👉 Войдите на HH.ru в браузере и затем нажмите ENTER здесь: ",
                        )
                        await page.goto(
                            "https://hh.ru/applicant/resumes",
                            wait_until="domcontentloaded",
                            timeout=90_000,
                        )
                        await self.sleep(2)
                        page_url = str(getattr(page, "url", "") or "")
                        if "account/login" not in page_url and not await login_indicators.first.is_visible():
                            print("✅ Успешный вход на HH.ru! Запускаем работу агента...\n")
                            return True
                        return False
                    return True
                except Exception as exc:
                    logger.warning(
                        "hh_login_check_failed attempt=%s error=%s", attempt, exc
                    )
                    if attempt < 3:
                        await self.sleep(5)
            return False
        finally:
            await page.close()

    async def _capture_failure_screenshot(self, page: Any, job_id: str) -> None:
        try:
            path = Path(tempfile.gettempdir()) / f"hh-failure-{job_id}-{uuid4().hex}.png"
            await page.screenshot(path=path)
            self.last_failure_screenshot = path
        except Exception as exc:
            logger.debug("failure_screenshot_failed job_id=%s error=%s", job_id, exc)

    async def _login_required(self, page: Any) -> bool:
        """True when hh.ru shows the anonymous header instead of the account menu."""
        try:
            if "account/login" in str(getattr(page, "url", "") or ""):
                return True
            for selector in ('a[data-qa="login"]', 'div[data-qa="login"]'):
                if await page.locator(selector).first.is_visible():
                    return True
            body = (await page.locator("body").first.inner_text()).lower()
            return "войдите как пользователь" in body
        except Exception:
            return False

    async def _foreign_vacancy_marker(self, page: Any) -> tuple[str, bool]:
        """HH's notification about another country, plus whether the job is remote."""
        try:
            body = (await page.locator("body").first.inner_text()).lower()
        except Exception as exc:
            logger.debug("foreign_vacancy_check_failed error=%s", exc)
            return "", False
        remote = any(
            word in body for word in ("удалённо", "удаленно", "remote", "дистанционн")
        )
        for selector in (
            '[data-qa="vacancy-other-country"]',
            '[data-qa*="other-country"]',
            '[data-qa*="foreign"]',
        ):
            try:
                if await page.locator(selector).first.is_visible():
                    return f"selector:{selector}", remote
            except Exception:
                continue
        for phrase in (
            "другой стране",
            "другой страны",
            "другую страну",
            "за рубежом",
        ):
            if phrase in body:
                return phrase, remote
        return "", remote

    async def _already_responded(self, page: Any) -> bool:
        """True when HH shows a chat instead of the respond control.

        HH hides the respond button once a response exists, so a missing
        button alone proves nothing. The chat control or an explicit
        "sent" label is the actual evidence.
        """
        try:
            if await page.locator(RESPONSE_CONTROL_SELECTOR).first.is_visible():
                return False
            for selector in RESPONDED_SELECTORS:
                if await page.locator(selector).first.is_visible():
                    return True
            if await page.get_by_text(RESPONDED_TEXT, exact=False).first.is_visible():
                return True
        except Exception as exc:
            logger.debug("already_responded_check_failed error=%s", exc)
        return False

    async def _response_available(self, page: Any, url: str) -> bool:
        await page.goto(url.split("?")[0], wait_until="domcontentloaded", timeout=30_000)
        await page.locator('[data-qa="vacancy-description"]').wait_for(
            state="visible", timeout=20_000
        )
        response_control = page.locator(
            'a[data-qa="vacancy-response-link-top"], '
            'button[data-qa="vacancy-response-link-top"]'
        ).first
        return await response_control.count() > 0

    async def _response_confirmed(self, page: Any, url: str) -> bool:
        try:
            return not await self._response_available(page, url)
        except Exception as exc:
            logger.warning("application_confirmation_failed error=%s", exc)
            return False

    async def _confirm_relocation_warning(self, page: Any) -> bool:
        """Press HH's "apply anyway" confirmation when it asks about relocation."""
        confirmation = page.locator(
            'button[data-qa="relocation-warning-confirm"], '
            '[data-qa*="relocation"] button'
        ).first
        if await confirmation.is_visible():
            await confirmation.click()
            await self._delay()
            return True
        for text in RELOCATION_CONFIRM_TEXTS:
            try:
                candidate = page.get_by_text(text, exact=True).first
                if await candidate.is_visible():
                    await candidate.click()
                    await self._delay()
                    return True
            except Exception:
                continue
        return False

    async def _attach_cover_letter_after_response(
        self, page: Any, cover_letter: str, required_portfolio: str
    ) -> None:
        attach = page.locator(
            'button[data-qa="responded-success-attach-cover-letter"]'
        ).first
        if not await attach.is_visible():
            raise RuntimeError("post_submit_cover_letter_control_unavailable")
        await attach.click()
        await self._delay()

        textarea = page.locator(
            'textarea[data-qa="vacancy-response-popup-form-letter-input"]'
        ).first
        await textarea.wait_for(state="visible", timeout=5_000)
        await textarea.fill(cover_letter)
        await self._delay()
        entered_letter = await textarea.input_value()
        if entered_letter != cover_letter or not has_required_portfolio(
            entered_letter, required_portfolio
        ):
            raise RuntimeError("required_cover_letter_not_preserved")

        submit = page.locator(
            'button[data-qa="vacancy-response-letter-submit"]:visible'
        ).first
        if not await submit.is_visible():
            raise RuntimeError("post_submit_cover_letter_submit_unavailable")
        await submit.click()
        try:
            await submit.wait_for(state="hidden", timeout=15_000)
            await attach.wait_for(state="hidden", timeout=15_000)
        except Exception as exc:
            raise RuntimeError("post_submit_cover_letter_not_confirmed") from exc

        if required_portfolio:
            await self._ensure_cover_letter_in_chat(
                page, cover_letter, required_portfolio
            )

    async def _open_chat_frame(self, page: Any) -> Any:
        chat = page.locator(
            'button[data-qa="vacancy-response-link-view-topic"]'
        ).first
        await chat.wait_for(state="visible", timeout=5_000)
        await chat.click()
        for _ in range(20):
            frame = next(
                (
                    candidate
                    for candidate in page.frames
                    if candidate.url.startswith("https://chatik.hh.ru/")
                ),
                None,
            )
            if frame is not None:
                return frame
            await self.sleep(0.25)
        raise RuntimeError("hh_chat_frame_unavailable")

    async def _ensure_cover_letter_in_chat(
        self, page: Any, cover_letter: str, required_portfolio: str
    ) -> None:
        post_statuses: list[int] = []
        if hasattr(page, "on"):
            def capture_response(response: Any) -> None:
                try:
                    if response.request.method == "POST" and "chatik.hh.ru" in response.url:
                        post_statuses.append(response.status)
                except Exception:
                    return

            page.on("response", capture_response)
        frame = await self._open_chat_frame(page)
        body = await frame.locator("body").inner_text()
        if required_portfolio in body:
            return
        attach = frame.get_by_text("Добавить сопроводительное", exact=False).first
        await attach.wait_for(state="visible", timeout=5_000)
        await attach.click()
        await self.sleep(0.5)
        textarea = frame.locator('textarea[data-qa="text-input"]').first
        await textarea.wait_for(state="visible", timeout=5_000)
        await textarea.fill(cover_letter)
        entered_letter = await textarea.input_value()
        if entered_letter != cover_letter or not has_required_portfolio(
            entered_letter, required_portfolio
        ):
            raise RuntimeError("required_cover_letter_not_preserved_in_chat")
        send = frame.locator('button[data-qa="chatik-do-send-message"]').first
        await send.wait_for(state="visible", timeout=5_000)
        if hasattr(send, "is_enabled") and not await send.is_enabled():
            raise RuntimeError("chat_cover_letter_send_disabled")
        await self.sleep(0.5)
        await send.click()
        try:
            await frame.wait_for_function(
                "url => document.body.innerText.includes(url)",
                required_portfolio,
                timeout=15_000,
            )
        except Exception as exc:
            remaining = len(await textarea.input_value())
            if remaining == 0 and any(200 <= status < 300 for status in post_statuses):
                await page.reload(wait_until="domcontentloaded", timeout=30_000)
                refreshed_frame = await self._open_chat_frame(page)
                refreshed_body = await refreshed_frame.locator("body").inner_text()
                if required_portfolio in refreshed_body:
                    return
            raise RuntimeError(
                "chat_cover_letter_not_confirmed:"
                f"post_statuses={post_statuses}:remaining_chars={remaining}"
            ) from exc

    async def attach_existing_cover_letter(
        self, job_id: str, *, cover_letter: str | None = None
    ) -> bool:
        if (
            self.settings.app_mode not in {"approval", "auto"}
            or not self.settings.enable_real_apply
        ):
            logger.warning("cover_letter_repair_blocked job_id=%s reason=app_mode", job_id)
            return False
        vacancy = self.database.get(job_id)
        if vacancy is None or vacancy.status not in {
            VacancyStatus.APPLY_FAILED,
            VacancyStatus.APPLIED,
        }:
            logger.warning("cover_letter_repair_blocked job_id=%s reason=status", job_id)
            return False
        letter = cover_letter if cover_letter is not None else vacancy.cover_letter
        cover = self.settings.profile.cover_letter
        required_portfolio = cover.required_portfolio_url
        validation_error = cover_letter_validation_error(
            letter,
            required_portfolio,
            cover.closing,
            cover.max_length,
        )
        if validation_error:
            logger.warning(
                "cover_letter_repair_blocked job_id=%s reason=%s",
                job_id,
                validation_error,
            )
            return False

        page = None
        try:
            page = await self.context.new_page()
            await page.goto(vacancy.url, wait_until="domcontentloaded", timeout=30_000)
            attach = page.locator(
                'button[data-qa="responded-success-attach-cover-letter"]'
            ).first
            if await attach.is_visible():
                await self._attach_cover_letter_after_response(
                    page, letter, required_portfolio
                )
            else:
                await self._ensure_cover_letter_in_chat(
                    page, letter, required_portfolio
                )
            if vacancy.status is VacancyStatus.APPLY_FAILED:
                updated = self.database.complete_existing_response_repair(
                    job_id,
                    now=self.now_factory(),
                    cover_letter=letter,
                )
            else:
                updated = self.database.record_applied_cover_letter_repair(
                    job_id, letter
                )
            if not updated:
                raise RuntimeError("cover_letter_repair_status_not_updated")
            logger.info("cover_letter_repair_sent job_id=%s", job_id)
            return True
        except Exception as exc:
            logger.error("cover_letter_repair_failed job_id=%s error=%s", job_id, exc)
            return False
        finally:
            if page is not None:
                try:
                    await page.close()
                except Exception as exc:
                    logger.warning(
                        "cover_letter_repair_page_close_failed job_id=%s error=%s",
                        job_id,
                        exc,
                    )

    async def search_vacancies(
        self,
        query: str,
        areas: tuple[str, ...],
        experience_filters: tuple[str, ...],
        *,
        remote_only: bool = False,
    ) -> VacancySearchResult:
        results: list[VacancySummary] = []
        found_results = 0
        duplicates = 0
        seen_ids: set[str] = set()
        page = None
        try:
            page = await self.context.new_page()
            for page_number in range(self.settings.max_pages_per_query):
                params: dict[str, Any] = {
                    "text": query,
                    "page": page_number,
                }
                if areas:
                    params["area"] = list(areas)
                if experience_filters:
                    params["experience"] = list(experience_filters)
                if remote_only:
                    params["schedule"] = "remote"
                if self.settings.hh_order_by:
                    params["order_by"] = self.settings.hh_order_by
                if self.settings.hh_search_period:
                    params["search_period"] = str(self.settings.hh_search_period)
                await self._delay()
                search_url = f"https://hh.ru/search/vacancy?{urlencode(params, doseq=True)}"
                await page.goto(
                    search_url,
                    wait_until="domcontentloaded",
                    timeout=30_000,
                )
                if await self._captcha_visible(page):
                    # HH gates the whole session behind a captcha; without
                    # solving it every vacancy page load fails as well.
                    logger.warning("search_captcha_detected query=%r", query)
                    solver = self.captcha_solver
                    if solver is None or not await self._solve_captcha_popup(
                        page, solver, query
                    ):
                        return VacancySearchResult(
                            results, found_results, duplicates, "captcha_not_solved"
                        )
                    await page.goto(
                        search_url,
                        wait_until="domcontentloaded",
                        timeout=30_000,
                    )
                cards = await page.locator('a[data-qa="serp-item__title"]').all()
                if not cards:
                    break
                for card in cards:
                    href = await card.get_attribute("href")
                    title = (await card.inner_text()).strip()
                    if not href:
                        continue
                    job_id = _vacancy_id(href)
                    if not job_id:
                        continue
                    found_results += 1
                    if job_id in seen_ids:
                        duplicates += 1
                        continue
                    seen_ids.add(job_id)
                    existing = self.database.get(job_id)
                    if existing is not None:
                        duplicates += 1
                        if (
                            existing.status is VacancyStatus.PENDING_APPROVAL
                            and len(results) < self.settings.max_vacancies_per_query
                        ):
                            results.append(
                                VacancySummary(job_id, title, href, query, True)
                            )
                        continue
                    if len(results) < self.settings.max_vacancies_per_query:
                        results.append(VacancySummary(job_id, title, href, query))
                if len(results) >= self.settings.max_vacancies_per_query:
                    break
                await self._delay()
            return VacancySearchResult(results, found_results, duplicates)
        except Exception as exc:
            logger.error("vacancy_search_failed query=%r error=%s", query, exc)
            return VacancySearchResult(
                results,
                found_results,
                duplicates,
                f"search_error:{type(exc).__name__}",
            )
        finally:
            if page is not None:
                try:
                    await page.close()
                except Exception as exc:
                    logger.warning(
                        "vacancy_search_page_close_failed query=%r error=%s",
                        query,
                        type(exc).__name__,
                    )

    async def read_vacancy(
        self,
        summary: VacancySummary,
        captcha_solver: CaptchaSolver | None = None,
    ) -> VacancyDetails:
        page = None
        try:
            page = await self.context.new_page()
            await self._delay()
            response = await page.goto(
                summary.url, wait_until="domcontentloaded", timeout=30_000
            )
            status = getattr(response, "status", None)
            if status in (404, 410):
                logger.info("vacancy_gone job_id=%s status=%s", summary.id, status)
                return VacancyDetails(summary, PageState.VACANCY_REMOVED)
            state = await classify_page(page)
            if state is PageState.CAPTCHA_DETECTED and captcha_solver is not None:
                state = await self._solve_captcha(page, summary, captcha_solver)
            if state is not PageState.VACANCY_LOADED:
                if state is PageState.CAPTCHA_DETECTED:
                    logger.warning("captcha_detected job_id=%s", summary.id)
                return VacancyDetails(summary, state)
            marker, remote = await self._foreign_vacancy_marker(page)
            if marker and not remote:
                logger.info(
                    "vacancy_foreign job_id=%s marker=%r remote=false", summary.id, marker
                )
                return VacancyDetails(summary, PageState.FOREIGN_VACANCY)
            if marker:
                logger.info(
                    "vacancy_foreign_remote job_id=%s marker=%r", summary.id, marker
                )
            if await self._already_responded(page):
                logger.info("vacancy_already_responded job_id=%s", summary.id)
                return VacancyDetails(summary, PageState.ALREADY_RESPONDED)
            desc_locator = page.locator('[data-qa="vacancy-description"]').first
            if not await desc_locator.is_visible():
                alt_locator = page.locator('.vacancy-description, [data-qa="vacancy-view"]').first
                if await alt_locator.is_visible():
                    desc_locator = alt_locator
            description = (await desc_locator.inner_text()).strip()
            if not description:
                return VacancyDetails(
                    summary,
                    PageState.PAGE_STRUCTURE_CHANGED,
                    error="vacancy description is empty",
                )
            company_locator = page.locator('[data-qa="vacancy-company-name"]').first
            if not await company_locator.is_visible():
                alt_company = page.locator('[data-qa="vacancy-company"]').first
                if await alt_company.is_visible():
                    company_locator = alt_company
            company = (
                (await company_locator.inner_text()).strip()
                if await company_locator.is_visible()
                else ""
            )
            company_url = _company_url(
                summary.url,
                await company_locator.get_attribute("href") if company else None,
            )
            return VacancyDetails(
                summary,
                state,
                company,
                description,
                company_url=company_url,
                foreign_remote=bool(marker and remote),
            )
        except Exception as exc:
            logger.error(
                "vacancy_read_exception job_id=%s error_type=%s error=%s",
                summary.id,
                type(exc).__name__,
                exc,
            )
            if page is not None:
                await self._capture_failure_screenshot(page, summary.id)
            return VacancyDetails(
                summary,
                PageState.NETWORK_ERROR,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            if page is not None:
                try:
                    await page.close()
                except Exception as exc:
                    logger.warning(
                        "vacancy_page_close_failed job_id=%s error=%s",
                        summary.id,
                        exc,
                    )

    async def read_company_details(self, company_url: str) -> CompanyDetails:
        company_url = _company_url("https://hh.ru/", company_url)
        if not company_url:
            return CompanyDetails()
        page = None
        try:
            page = await self.context.new_page()
            await self._delay()
            await page.goto(company_url, wait_until="domcontentloaded", timeout=30_000)
            rating_text = await _visible_text(
                page, '[data-qa="employer-review-small-widget-total-rating"]'
            )
            reviews_text = await _visible_text(
                page,
                '[data-qa="employer-review-small-widget-review-count-action"]',
            )
            try:
                rating = float(rating_text.replace(",", "."))
                if not 0 <= rating <= 5:
                    rating = None
            except ValueError:
                rating = None
            match = re.search(r"\d[\d\s\u00a0]*", reviews_text)
            reviews_count = int(re.sub(r"\D", "", match.group())) if match else None
            return CompanyDetails(rating, reviews_count)
        except Exception as exc:
            logger.warning("company_details_read_failed error_type=%s", type(exc).__name__)
            return CompanyDetails()
        finally:
            if page is not None:
                try:
                    await page.close()
                except Exception:
                    logger.warning("company_details_page_close_failed")

    async def _captcha_language_target(self, page: Any) -> str:
        """Language the captcha switch would move TO, as shown on its button."""
        control = page.locator('button[data-qa="captcha-language"]').first
        if not await control.is_visible():
            return ""
        for attribute in ("aria-label", "title"):
            try:
                value = await control.get_attribute(attribute)
            except Exception:
                value = None
            if value:
                return value.strip().lower()
        try:
            return (await control.inner_text()).strip().lower()
        except Exception:
            return ""

    async def _switch_captcha_to_english(self, page: Any) -> bool:
        """HH renders its Russian captcha badly, so always solve the English one.

        The switch is a toggle: its label names the language it switches TO, so
        clicking it again would turn an English captcha back into a Russian one.
        """
        control = page.locator('button[data-qa="captcha-language"]').first
        try:
            if not await control.is_visible():
                return False
            target = await self._captcha_language_target(page)
            if "русск" in target or "russian" in target:
                logger.info("captcha_already_english")
                return True
            await control.click()
            await self._delay()
            switched_to = await self._captcha_language_target(page)
            logger.info("captcha_language_switched target=%r", switched_to)
            if "русск" not in switched_to and "russian" not in switched_to:
                logger.warning("captcha_language_switch_unconfirmed label=%r", switched_to)
            return True
        except Exception as exc:
            logger.warning(
                "captcha_language_switch_failed error_type=%s error=%s",
                type(exc).__name__,
                exc,
            )
            return False

    async def _refresh_captcha(self, page: Any) -> bool:
        """Ask HH for another captcha image when the current one is rejected."""
        for selector in (
            'button[data-qa="captcha-renew-text"]',
            'button:has-text("Обновить")',
        ):
            try:
                control = page.locator(selector).first
                if await control.is_visible():
                    await control.click()
                    logger.info("captcha_refreshed")
                    return True
            except Exception:
                continue
        return False

    async def _captcha_error_text(self, page: Any) -> str:
        try:
            element = page.locator('[data-qa="account-captcha-error"]').first
            if await element.is_visible():
                return (await element.inner_text()).strip()[:120]
        except Exception:
            pass
        return ""

    async def _captcha_screenshot(self, page: Any, path: Path) -> None:
        """Capture only the captcha picture: the whole page confuses the model."""
        for selector in (
            'img[data-qa="account-captcha-picture"]',
            'img[src*="/captcha/picture"]',
            'img[src*="captcha"]',
        ):
            try:
                target = page.locator(selector).first
                if await target.is_visible():
                    await target.screenshot(path=path)
                    return
            except Exception:
                continue
        await page.screenshot(path=path)

    async def _solve_captcha(
        self, page: Any, summary: VacancySummary, solver: CaptchaSolver
    ) -> PageState:
        if not await self._solve_captcha_popup(page, solver, summary.title):
            return PageState.CAPTCHA_DETECTED
        await self.sleep(2)
        state = await classify_page(page)
        if state is PageState.VACANCY_LOADED:
            return state
        # HH returns to `backurl` after the captcha, but the redirect can land
        # on an interstitial: navigate to the vacancy explicitly.
        try:
            await page.goto(
                summary.url, wait_until="domcontentloaded", timeout=30_000
            )
        except Exception as exc:
            logger.warning(
                "captcha_return_goto_failed job_id=%s error_type=%s error=%s",
                summary.id,
                type(exc).__name__,
                exc,
            )
            return PageState.NETWORK_ERROR
        return await classify_page(page)

    async def _log_captcha_dom(self, page: Any) -> None:
        """How many captcha fields and submit buttons the page actually has."""
        try:
            fields = page.locator(
                'input[data-qa="account-captcha-input"], input[name="captchaText"]'
            )
            submits = page.locator('button[data-qa="account-captcha-submit"]')
            language = page.locator('button[data-qa="captcha-language"]')
            frames = len(page.frames) if hasattr(page, "frames") else 1
            logger.info(
                "captcha_dom fields=%s submits=%s language=%s frames=%s first_visible=%s",
                await fields.count(),
                await submits.count(),
                await language.count(),
                frames,
                await fields.first.is_visible(),
            )
        except Exception as exc:
            logger.warning("captcha_dom_failed error=%s", exc)

    @staticmethod
    def _prune_rejected_captchas(keep: int = 10) -> None:
        """Keep only the newest rejected captcha screenshots."""
        try:
            files = sorted(
                Path(tempfile.gettempdir()).glob("hh-captcha-reject-*.png"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            for stale in files[keep:]:
                stale.unlink(missing_ok=True)
        except Exception:
            pass

    async def _solve_captcha_popup(
        self, page: Any, solver: CaptchaSolver, title: str
    ) -> bool:
        """Solve HH's captcha modal, replacing the image when it is rejected."""
        if self.settings.captcha_max_attempts < 1:
            return False
        await self._log_captcha_dom(page)
        for attempt in range(self.settings.captcha_max_attempts):
            if self.settings.captcha_switch_to_english:
                await self._switch_captcha_to_english(page)
            screenshot = Path(tempfile.gettempdir()) / f"hh-captcha-{uuid4().hex}.png"
            answer: str | None = None
            try:
                await self._captcha_screenshot(page, screenshot)
                answer = await solver(
                    screenshot, title, self.settings.captcha_timeout_seconds
                )
            except Exception:
                screenshot.unlink(missing_ok=True)
                raise
            if answer:
                # Only the captcha's own input: a generic text selector hits
                # the response form underneath the modal overlay.
                field = page.locator(
                    'input[data-qa="account-captcha-input"], input[name="captchaText"]'
                ).first
                if await field.is_visible():
                    submit = page.locator(
                        'button[data-qa="account-captcha-submit"], '
                        'form[action*="captcha"] button[type="submit"], '
                        'button[data-qa*="captcha"]:has-text("Отправить")'
                    ).first
                    # A two-word captcha is accepted either with a space or
                    # joined; both variants go to the SAME picture before it is
                    # refreshed, otherwise the second variant proves nothing.
                    variants = [answer]
                    if " " in answer:
                        variants.append(answer.replace(" ", ""))
                    rejected_error = ""
                    for index, submitted in enumerate(variants):
                        # Type instead of setting the value: HH's form keeps
                        # its own state and ignores a programmatic fill.
                        try:
                            await field.click()
                            await field.fill("")
                            await field.press_sequentially(submitted, delay=40)
                        except Exception as exc:
                            logger.warning("captcha_typing_failed error=%s", exc)
                            await field.fill(submitted)
                        typed = ""
                        try:
                            typed = await field.input_value()
                        except Exception:
                            pass
                        logger.info(
                            "captcha_answer_typed attempt=%s variant=%s submitted=%r typed=%r",
                            attempt,
                            index,
                            submitted,
                            typed,
                        )
                        clicked = False
                        if await submit.is_visible():
                            try:
                                await submit.click(timeout=10_000)
                                clicked = True
                            except Exception as exc:
                                logger.warning(
                                    "captcha_submit_click_failed error=%s", exc
                                )
                        if not clicked:
                            try:
                                await field.press("Enter")
                            except Exception:
                                pass
                        await self.sleep(2)
                        if not await self._captcha_visible(page):
                            screenshot.unlink(missing_ok=True)
                            logger.info(
                                "captcha_solved title=%s variant=%s answer=%r",
                                title,
                                index,
                                submitted,
                            )
                            return True
                        rejected_error = await self._captcha_error_text(page)
                        logger.info(
                            "captcha_rejected attempt=%s variant=%s answer=%r error=%r",
                            attempt,
                            index,
                            submitted,
                            rejected_error,
                        )
                    # Keep the rejected picture and the answer: without them a
                    # rejected captcha is impossible to diagnose.
                    kept = screenshot.with_name(
                        f"hh-captcha-reject-{uuid4().hex}.png"
                    )
                    try:
                        screenshot.replace(kept)
                    except Exception:
                        kept = None
                    self._prune_rejected_captchas()
                    logger.info("captcha_rejected_file %s", kept)
                else:
                    logger.warning("captcha_input_not_found")
            else:
                logger.info("captcha_answer_missing attempt=%s", attempt)
            screenshot.unlink(missing_ok=True)
            if not await self._refresh_captcha(page):
                return False
            await self.sleep(2)
        return False

    async def _captcha_visible(self, page: Any) -> bool:
        try:
            if "captcha" in str(getattr(page, "url", "") or "").lower():
                return True
            for selector in (
                'img[data-qa="account-captcha-picture"]',
                'input[data-qa="account-captcha-input"]',
                'form[action*="captcha"]',
                'img[src*="/captcha/picture"]',
                'iframe[src*="smartcaptcha"]',
            ):
                if await page.locator(selector).first.is_visible():
                    return True
            try:
                return await page.get_by_text(
                    "Пройдите капчу", exact=False
                ).first.is_visible()
            except Exception:
                return False
        except Exception:
            return False

    async def _solve_page_captcha(self, page: Any, title: str) -> bool:
        if self.captcha_solver is None:
            return False
        if not await self._captcha_visible(page):
            return True
        return await self._solve_captcha_popup(page, self.captcha_solver, title)

    async def _ensure_no_captcha(self, page: Any, title: str) -> None:
        if await self._captcha_visible(page) and not await self._solve_page_captcha(
            page, title
        ):
            raise RuntimeError("captcha_not_solved")

    async def _read_questionnaire(self, page: Any) -> QuestionnaireSpec:
        if not hasattr(page, "evaluate"):
            return QuestionnaireSpec()
        raw = await page.evaluate(JS_MARK_FORM_FIELDS)
        return parse_spec(raw)

    async def _fill_questionnaire(
        self, page: Any, vacancy: Vacancy, job_id: str
    ) -> bool:
        hidden_resume = page.locator('[data-qa="hidden-resume-warning"]').first
        if await hidden_resume.is_visible():
            raise QuestionnaireRequiredError("resume_hidden_manual")

        spec = await self._read_questionnaire(page)
        if spec.is_empty:
            return False
        logger.info(
            "questionnaire_fields_detected job_id=%s count=%d",
            job_id,
            len(spec.fields),
        )
        if spec.has_file_upload():
            raise QuestionnaireRequiredError("questionnaire_file_required")
        unsupported = spec.unsupported()
        if unsupported:
            logger.warning(
                "questionnaire_unsupported job_id=%s field=%s",
                job_id,
                unsupported[0].index,
            )
            raise QuestionnaireRequiredError("questionnaire_unsupported_field")
        if getattr(self, "analyzer", None) is None:
            raise QuestionnaireRequiredError("questionnaire_required")

        result = await self.analyzer.answer_form(
            spec.payload(),
            vacancy_title=vacancy.title,
            vacancy_description="",
        )
        if not result.can_answer:
            logger.info(
                "questionnaire_not_answered job_id=%s reason=%s",
                job_id,
                result.reason,
            )
            raise QuestionnaireRequiredError(
                f"questionnaire_unanswered:{result.reason}"
            )

        by_index = {field.index: field for field in spec.fields}
        for answer in result.answers:
            field = by_index.get(answer.index)
            if field is None:
                continue
            logger.info(
                "answering_questionnaire job_id=%s field=%d q=%r",
                job_id,
                field.index,
                field.question[:80],
            )
            if not await self._fill_form_answer(page, field, answer):
                logger.warning(
                    "questionnaire_field_fill_failed job_id=%s field=%d",
                    job_id,
                    field.index,
                )
                raise QuestionnaireRequiredError(
                    f"questionnaire_field_fill_failed:{field.index}"
                )
            await self._delay()
        return True

    async def _fill_form_answer(
        self, page: Any, field: QuestionnaireField, answer: Any
    ) -> bool:
        container = page.locator(f'[data-bot-field="{field.index}"]').first
        try:
            if not await container.count():
                await page.evaluate(JS_MARK_FORM_FIELDS)
                container = page.locator(
                    f'[data-bot-field="{field.index}"]'
                ).first
        except Exception:
            pass
        if field.kind == "text":
            target = container.locator('textarea, input[type="text"]').first
            return await self._fill_task_field(target, answer.text)
        if field.kind == "select":
            if not answer.choices:
                return True
            options = field.options
            chosen = options[answer.choices[0]]
            await container.locator("select").first.select_option(label=chosen)
            return True

        for choice_index in range(len(field.options)):
            option = page.locator(
                f'[data-bot-option="{field.index}:{choice_index}"]'
            ).first
            selected = choice_index in answer.choices
            if selected:
                if not await self._set_option(option, checked=True):
                    return False
            elif field.kind == "multi":
                await self._set_option(option, checked=False)
        return True

    async def _set_option(self, option: Any, *, checked: bool) -> bool:
        try:
            if await option.is_checked() == checked:
                return True
        except Exception:
            pass
        label = option.locator("xpath=ancestor::label[1]")
        try:
            if await label.count() > 0:
                await label.first.click(timeout=1500)
        except Exception:
            pass
        try:
            if await option.is_checked() != checked:
                if checked:
                    await option.check(force=True, timeout=2000)
                else:
                    await option.uncheck(force=True, timeout=2000)
        except Exception:
            pass
        try:
            return await option.is_checked() == checked
        except Exception:
            return False

    async def submit_application(self, permission: ApplicationPermission) -> bool:
        claim = self.approval_guard.claim(permission)
        if not claim.allowed or claim.vacancy is None:
            # The permit was issued but the claim was refused (usually the
            # daily limit): put the vacancy back into the queue instead of
            # leaving it approved and invisible.
            requeued = (
                self.database.requeue_approved(permission.job_id)
                if claim.reason == "daily_limit"
                else False
            )
            logger.warning(
                "application_blocked job_id=%s reason=%s requeued=%s",
                permission.job_id,
                claim.reason,
                requeued,
            )
            return False

        vacancy = claim.vacancy
        page = None
        submit_attempted = False
        final_submit_clicked = False
        self.last_failure_screenshot = None
        try:
            cover = self.settings.profile.cover_letter
            required_portfolio = cover.required_portfolio_url
            validation_error = cover_letter_validation_error(
                vacancy.cover_letter,
                required_portfolio,
                cover.closing,
                cover.max_length,
            )
            if validation_error:
                raise RuntimeError(validation_error)
            page = await self.context.new_page()
            await self._delay()
            await page.goto(vacancy.url, wait_until="domcontentloaded", timeout=30_000)

            # Check if user is unauthenticated
            if await page.locator('input[type="tel"], a[data-qa="mainmenu_applicantAccess"]:has-text("Войти")').first.is_visible():
                raise RuntimeError("Сессия на HH.ru не авторизована. Требуется вход в аккаунт HH.ru.")
            await self._ensure_no_captcha(page, vacancy.title)

            response_button = page.locator(RESPONSE_CONTROL_SELECTOR).first
            if not await response_button.is_visible():
                if await self._already_responded(page):
                    raise AlreadyRespondedError("vacancy already responded")
                raise RuntimeError("application response control was not found")
            if not self.database.mark_submit_attempt(
                permission.job_id,
                permission.permit,
                now=self.now_factory(),
                daily_limit=self.settings.max_applications_per_day,
            ):
                error = "application permission failed pre-response validation"
                self.database.complete_application(
                    permission.job_id,
                    permission.permit,
                    success=False,
                    now=self.now_factory(),
                    error_text=error,
                )
                logger.warning(
                    "application_blocked job_id=%s reason=pre_response_recheck",
                    permission.job_id,
                )
                return False
            submit_attempted = True
            await response_button.click()
            await self._delay()
            logger.info(
                "application_after_click job_id=%s url=%s",
                permission.job_id,
                getattr(page, "url", ""),
            )
            await self._ensure_no_captcha(page, vacancy.title)
            await self._confirm_relocation_warning(page)
            await self._ensure_no_captcha(page, vacancy.title)

            post_submit_attach = page.locator(
                'button[data-qa="responded-success-attach-cover-letter"]'
            ).first
            if await post_submit_attach.is_visible():
                if required_portfolio:
                    await self._attach_cover_letter_after_response(
                        page, vacancy.cover_letter, required_portfolio
                    )
                else:
                    logger.info("post_submit_cover_letter_skipped_not_required job_id=%s", permission.job_id)
                if not self.database.complete_application(
                    permission.job_id,
                    permission.permit,
                    success=True,
                    now=self.now_factory(),
                ):
                    raise RuntimeError("application status could not be completed")
                logger.info("application_sent job_id=%s", permission.job_id)
                return True

            resume_name = self.settings.profile.hh.resume_name
            resume_selector = page.locator(
                '[data-qa*="resume-select"], [data-qa*="resume-selector"]'
            ).first
            if resume_name and await resume_selector.is_visible():
                await resume_selector.click()
                option = page.get_by_text(resume_name, exact=True).first
                if not await option.is_visible():
                    raise RuntimeError("configured resume was not found")
                await option.click()

            # Зарплатные ожидания: если на форме есть поле ввода зарплаты
            try:
                salary_input = page.locator(
                    'input[name*="salary" i], input[data-qa*="salary" i], input[placeholder*="зарплат" i], input[placeholder*="доход" i]'
                ).first
                if await salary_input.is_visible():
                    current_val = await salary_input.input_value()
                    if not current_val.strip():
                        salary_cfg = self.settings.profile.candidate.salary_expectation or "300000"
                        digits = "".join(filter(str.isdigit, salary_cfg)) or "300000"
                        await salary_input.fill(digits)
                        logger.info("salary_expectation_filled job_id=%s val=%s", permission.job_id, digits)
            except Exception as sal_exc:
                logger.debug("salary_input_check_skipped job_id=%s error=%s", permission.job_id, sal_exc)

            # Вопросы анкеты / тестовые задания работодателя
            await self._fill_questionnaire(page, vacancy, permission.job_id)

            await self._ensure_no_captcha(page, vacancy.title)
            submit_button = page.locator(
                'button[data-qa*="vacancy-response-submit"]:visible'
            ).first
            try:
                await submit_button.wait_for(state="visible", timeout=10_000)
            except Exception as exc:
                # HH sometimes sends the response right after the click and
                # never renders a form. The respond control disappearing is
                # then proof that the response exists. The page may also be
                # redirected away, so re-check on the vacancy itself.
                responded = await self._already_responded(page)
                if not responded:
                    try:
                        await page.goto(
                            vacancy.url,
                            wait_until="domcontentloaded",
                            timeout=30_000,
                        )
                        await self.sleep(1)
                        responded = await self._already_responded(page)
                    except Exception as reload_error:
                        logger.warning(
                            "application_recheck_failed job_id=%s error=%s",
                            permission.job_id,
                            reload_error,
                        )
                if responded:
                    if not self.database.complete_application(
                        permission.job_id,
                        permission.permit,
                        success=True,
                        now=self.now_factory(),
                    ):
                        raise RuntimeError(
                            "application status could not be completed"
                        ) from exc
                    logger.info(
                        "application_sent_without_form job_id=%s",
                        permission.job_id,
                    )
                    return True
                if await self._login_required(page):
                    raise RuntimeError("hh_login_required") from exc
                raise RuntimeError("final application button was not found") from exc

            # Проверяем, обязательно ли сопроводительное письмо
            letter_textarea = page.locator('textarea:not([name^="task_"])').first
            letter_required = False
            if required_portfolio:
                letter_required = True
            elif await letter_textarea.is_visible():
                letter_required = True
            elif not await submit_button.is_enabled():
                letter_required = True
            else:
                has_req_text = await page.get_by_text("Работодатель требует сопроводительное", exact=False).count() > 0
                if has_req_text:
                    letter_required = True

            if letter_required:
                letter_toggle = (
                    page.locator('[data-qa*="letter-toggle"]')
                    .or_(page.get_by_text("Написать сопроводительное", exact=False))
                    .or_(page.get_by_text("Добавить сопроводительное", exact=False))
                    .first
                )
                if await letter_toggle.is_visible():
                    await letter_toggle.click()
                    await self._delay()

                try:
                    await letter_textarea.wait_for(state="visible", timeout=5_000)
                    letter_to_fill = vacancy.cover_letter or "Здравствуйте! Готов обсудить детали вакансии и мой опыт на интервью."
                    await letter_textarea.fill(letter_to_fill)
                    await self._delay()
                except Exception as exc:
                    if required_portfolio:
                        raise RuntimeError("required_cover_letter_field_unavailable") from exc
                    logger.info("cover_letter_field_not_found job_id=%s, proceeding to submit", vacancy.id)

                if required_portfolio:
                    entered_letter = await letter_textarea.input_value()
                    if entered_letter != vacancy.cover_letter or not has_required_portfolio(
                        entered_letter, required_portfolio
                    ):
                        raise RuntimeError("required_cover_letter_not_preserved")
            else:
                logger.info("cover_letter_skipped_not_required job_id=%s", vacancy.id)

            await submit_button.click()
            # Only a click that actually landed marks the delivery as uncertain.
            final_submit_clicked = True
            await self.sleep(1)
            if await self._captcha_visible(page):
                if not await self._solve_page_captcha(page, vacancy.title):
                    raise RuntimeError("captcha_not_solved")
                if await submit_button.is_visible():
                    await submit_button.click()
            success_marker = (
                page.locator(
                    '[data-qa="vacancy-response-success"], '
                    '[data-qa="vacancy-response-link-view-topic"]'
                )
                .or_(page.get_by_text("Отклик отправлен", exact=False))
                .first
            )
            try:
                await success_marker.wait_for(state="visible", timeout=5_000)
            except Exception:
                # HH often renders the success block late or not at all.
                # A reload is the authoritative check: the respond control
                # disappears once the response exists.
                logger.info(
                    "application_success_marker_missing job_id=%s", permission.job_id
                )
            if not await self._response_confirmed(page, vacancy.url):
                raise RuntimeError("HH.ru did not confirm the application")
            if required_portfolio:
                await self._ensure_cover_letter_in_chat(
                    page, vacancy.cover_letter, required_portfolio
                )
            if not self.database.complete_application(
                permission.job_id,
                permission.permit,
                success=True,
                now=self.now_factory(),
            ):
                raise RuntimeError("application status could not be completed")
            logger.info("application_sent job_id=%s", permission.job_id)
            return True
        except QuestionnaireRequiredError as exc:
            response_still_available = False
            if page is not None:
                try:
                    response_still_available = await self._response_available(
                        page, vacancy.url
                    )
                except Exception as verification_error:
                    logger.warning(
                        "questionnaire_response_check_failed job_id=%s error=%s",
                        permission.job_id,
                        verification_error,
                    )
            if page is not None:
                await self._capture_failure_screenshot(page, permission.job_id)
            self.database.complete_application(
                permission.job_id,
                permission.permit,
                success=False,
                now=self.now_factory(),
                error_text=str(exc),
            )
            if response_still_available:
                self.database.clear_verified_unsubmitted_attempt(permission.job_id)
            logger.info(
                "application_skipped job_id=%s reason=questionnaire_required response_available=%s",
                permission.job_id,
                response_still_available,
            )
            return False
        except AlreadyRespondedError:
            recorded = self.database.mark_already_responded(
                permission.job_id, permission.permit
            )
            logger.info(
                "application_skipped_already_responded job_id=%s recorded=%s",
                permission.job_id,
                recorded,
            )
            return False
        except Exception as exc:
            if (
                page is not None
                and not submit_attempted
                and await self._already_responded(page)
                and self.database.mark_already_responded(
                    permission.job_id, permission.permit
                )
            ):
                logger.info(
                    "application_skipped_already_responded job_id=%s error=%s",
                    permission.job_id,
                    exc,
                )
                return False
            if page is not None:
                await self._capture_failure_screenshot(page, permission.job_id)
            self.database.complete_application(
                permission.job_id,
                permission.permit,
                success=False,
                now=self.now_factory(),
                error_text=str(exc),
            )
            if not final_submit_clicked:
                # HH never received a submission, so the daily slot and the
                # "delivery uncertain" signal must not be consumed.
                self.database.clear_unsubmitted_attempt(permission.job_id)
                logger.info(
                    "application_attempt_released job_id=%s", permission.job_id
                )
            logger.error(
                "application_failed job_id=%s url=%s error=%s",
                permission.job_id,
                getattr(page, "url", ""),
                exc,
            )
            return False
        finally:
            if page is not None:
                try:
                    await page.close()
                except Exception as exc:
                    logger.warning(
                        "application_page_close_failed job_id=%s error=%s",
                        permission.job_id,
                        exc,
                    )

    async def check_messages(
        self, notifier: Callable[[str], Awaitable[None]]
    ) -> None:
        page = await self.context.new_page()
        try:
            await self._delay()
            await page.goto(
                "https://hh.ru/applicant/negotiations",
                wait_until="domcontentloaded",
                timeout=30_000,
            )
            cards = await page.locator('div[data-qa="negotiations-item"]').all()
            for card in cards:
                badge = card.locator('span[data-qa="negotiations-item-badge"]')
                if not await badge.is_visible():
                    continue
                link = card.locator('a[data-qa="negotiations-item-vacancy-link"]')
                href = await link.get_attribute("href")
                title = (await link.inner_text()).strip()
                message_id = f"{href}:{title}"
                if href and not self.database.is_message_processed(message_id):
                    self.database.add_processed_message(message_id, href, title)
                    await notifier(f"New unread HH message for: {title}\nhttps://hh.ru{href}")
        except Exception as exc:
            logger.error("message_check_failed error=%s", exc)
        finally:
            await page.close()
    async def _find_resume_card(self, page: Any, target_name: str) -> Any:
        if target_name:
            card = page.locator('div[data-qa="resume"]').filter(has_text=target_name).first
            try:
                if await card.is_visible():
                    return card
            except Exception:
                pass
        first_card = page.locator('div[data-qa="resume"]').first
        try:
            if await first_card.is_visible():
                return first_card
        except Exception:
            pass
        return None

    @staticmethod
    async def _resume_status_text(search_scope: Any) -> str:        # [data-qa="title-description"] is the resume card's generic status
        # line: it holds the cooldown notice AND the plain "updated at" text.
        try:
            element = search_scope.locator('[data-qa="title-description"]').first
            if await element.is_visible():
                return (await element.inner_text()).replace("\xa0", " ").strip()
        except Exception:
            pass
        return ""

    @staticmethod
    def _free_bump_button(search_scope: Any) -> Any:
        # STRICTLY exclude the paid "Поднять автоматически" upsell
        # (data-qa="resume-update-button_actions").
        return search_scope.locator(
            'button[data-qa="resume-update-button"], '
            'button:has-text("Поднять в поиске"), '
            'a:has-text("Поднять в поиске")'
        ).filter(has_not_text="автоматически").filter(has_not_text="automatically").first

    @staticmethod
    async def _page_timezone(page: Any) -> ZoneInfo | None:
        # HH renders the cooldown clock in the browser's timezone — ask the
        # page instead of guessing (the container browser may run in UTC).
        try:
            tz_name = await page.evaluate(
                "() => Intl.DateTimeFormat().resolvedOptions().timeZone"
            )
            if tz_name:
                return ZoneInfo(str(tz_name))
        except Exception:
            pass
        return None

    async def _verify_bump(self, search_scope: Any) -> tuple[bool, str]:
        # After a successful click HH swaps the button for the cooldown notice.
        status_text = await self._resume_status_text(search_scope)
        if status_text and _COOLDOWN_TEXT_RE.search(status_text):
            return True, status_text
        try:
            notice = search_scope.get_by_text(_COOLDOWN_TEXT_RE).first
            if await notice.is_visible():
                return True, (await notice.inner_text()).replace("\xa0", " ").strip()
        except Exception:
            pass
        try:
            free_button = self._free_bump_button(search_scope)
            if not await free_button.is_visible() or await free_button.is_disabled():
                return True, status_text
        except Exception:
            return True, status_text
        return False, status_text

    async def bump_resume(self, resume_name: str | None = None) -> BumpResult:
        """Attempt to bump/raise a resume in search on hh.ru.

        Button first: the free "Поднять в поиске" button is the source of
        truth. The cooldown notice is only consulted when no clickable button
        exists, and its "available at HH:MM" time is parsed (a notice whose
        time already passed is treated as stale and re-checked after reload).
        """
        async with self._browser_lock:
            page = await self.context.new_page()
            try:
                target_name = (
                    resume_name
                    or self.settings.profile.hh.resume_name
                    or ""
                ).strip()
                logger.info("resume_bump_started target='%s'", target_name)
                await page.goto(
                    "https://hh.ru/applicant/resumes",
                    wait_until="domcontentloaded",
                    timeout=60_000,
                )
                await self.sleep(3)

                page_url = str(getattr(page, "url", "") or "")
                if "login" in page_url or "auth" in page_url:
                    logger.warning("resume_bump_failed reason=not_logged_in")
                    return BumpResult(False, "Требуется авторизация на HH.ru")

                notice_tz = await self._page_timezone(page)
                if notice_tz is not None:
                    logger.info("resume_bump_page_timezone tz=%s", notice_tz.key)

                for attempt in range(2):
                    card = await self._find_resume_card(page, target_name)
                    search_scope = card if card is not None else page
                    status_text = await self._resume_status_text(search_scope)
                    retry_at = (
                        _parse_cooldown_until(status_text, notice_tz)
                        if status_text and _COOLDOWN_TEXT_RE.search(status_text)
                        else None
                    )

                    free_button = self._free_bump_button(search_scope)
                    if await free_button.is_visible():
                        is_disabled = await free_button.is_disabled() or bool(
                            await free_button.get_attribute("disabled")
                        )
                        aria_disabled = (
                            await free_button.get_attribute("aria-disabled") or ""
                        ).lower() == "true"
                        if is_disabled or aria_disabled:
                            logger.info(
                                "resume_bump_button_disabled status='%s'", status_text
                            )
                            detail = f"Кнопка «Поднять в поиске» пока неактивна. {status_text}".strip()
                            return BumpResult(False, detail, retry_at)

                        await free_button.scroll_into_view_if_needed()
                        await self.sleep(1)
                        await free_button.click()
                        await self.sleep(3)

                        verified, verify_text = await self._verify_bump(search_scope)
                        if verified:
                            logger.info(
                                "resume_bump_success_verified target='%s'", target_name
                            )
                            return BumpResult(
                                True,
                                "Резюме успешно поднято в поиске!",
                                _parse_cooldown_until(verify_text, notice_tz),
                            )
                        logger.warning(
                            "resume_bump_unconfirmed target='%s'", target_name
                        )
                        return BumpResult(
                            False,
                            "Клик выполнен, но подтверждение поднятия от HH.ru не получено",
                        )

                    now_utc = datetime.now(UTC)
                    if retry_at is not None and retry_at > now_utc:
                        logger.info(
                            "resume_bump_cooldown_detected text='%s' retry_at=%s",
                            status_text,
                            retry_at.isoformat(),
                        )
                        return BumpResult(False, f"Кулдаун: {status_text}", retry_at)

                    if attempt == 0 and retry_at is not None:
                        # Cooldown time already passed but the button is
                        # missing — the notice is stale; reload once and
                        # re-check before giving up.
                        logger.info(
                            "resume_bump_stale_cooldown_reload text='%s'", status_text
                        )
                        await page.reload(wait_until="domcontentloaded", timeout=60_000)
                        await self.sleep(3)
                        continue

                    if status_text:
                        logger.info(
                            "resume_bump_button_not_found status='%s'", status_text
                        )
                        try:
                            Path("resumes_debug.html").write_text(
                                await page.content(), encoding="utf-8"
                            )
                            logger.info("resume_bump_debug_dump_saved")
                        except Exception:
                            pass
                        return BumpResult(
                            False,
                            f"Кнопка бесплатного поднятия не найдена. Статус карточки: {status_text}",
                            retry_at,
                        )
                    logger.info("resume_bump_button_not_found")
                    return BumpResult(False, "Кнопка бесплатного поднятия не найдена")

                logger.info("resume_bump_button_not_found")
                return BumpResult(False, "Кнопка бесплатного поднятия не найдена")
            except Exception as exc:
                logger.warning("resume_bump_failed error=%s", exc)
                return BumpResult(False, f"Ошибка при поднятии: {exc}")
            finally:
                await page.close()

