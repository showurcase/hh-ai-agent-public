from __future__ import annotations

import asyncio
import html
import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
import re
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

from aiogram import Bot, Dispatcher, F
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InputRichBlockDetails,
    InputRichBlockDivider,
    InputRichBlockParagraph,
    InputRichBlockPreformatted,
    InputRichBlockSectionHeading,
    InputRichMessage,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    RichTextBold,
    BotCommand,
    RichTextUrl,
)

from approval import ApprovalService
from ai_priority import is_ai_llm_vacancy
from config import Settings
from cover_letter import has_required_portfolio
from database import Database, Vacancy
from fit_summary import FIT_SUMMARY_FALLBACK
from llm.errors import LLMError
from llm.mistral_keys import MistralKeyCheckResult, MistralKeyManager, MistralKeyView
from updater import ReleaseInfo, check_github_release
from version import __version__
from mascots import get_pesec_image, is_pesec_enabled

logger = logging.getLogger(__name__)
PRIVATE_REPLY = "This bot is private."
MISTRAL_KEY_INPUT_TTL = timedelta(minutes=15)


@dataclass
class AgentControl:
    paused: bool = False
    circuit_reason: str = ""
    consecutive_search_errors: int = 0
    next_run_at: datetime | None = None
    force_run: bool = False
    wake_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)




@dataclass
class MistralKeyInputSession:
    expires_at: datetime


class TelegramService:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        approval_service: ApprovalService,
        control: AgentControl,
        *,
        mistral_keys: MistralKeyManager | None = None,
        hh_client: Any | None = None,
        queue_page: Any | None = None,
        bot: Any | None = None,
        dispatcher: Dispatcher | None = None,
        now_factory: Callable[[], datetime] | None = None,
    ):
        self.settings = settings
        self.database = database
        self.approval_service = approval_service
        self.control = control
        self.mistral_keys = mistral_keys
        self.hh_client = hh_client
        self.queue_page = queue_page
        if bot is not None:
            self.bot = bot
        else:
            import os
            tg_proxy = os.getenv('TG_PROXY')
            if tg_proxy:
                from aiogram.client.session.aiohttp import AiohttpSession
                self.bot = Bot(token=settings.tg_bot_token, session=AiohttpSession(proxy=tg_proxy))
            else:
                self.bot = Bot(token=settings.tg_bot_token)
        self.dispatcher = dispatcher or Dispatcher()
        self.now_factory = now_factory or (lambda: datetime.now().astimezone())
        self._captcha_future: asyncio.Future[str | None] | None = None
        self._edit_job_id: str | None = None
        self._edit_future: asyncio.Future[str | None] | None = None
        self._mistral_key_input: MistralKeyInputSession | None = None
        self.latest_release: ReleaseInfo | None = None
        self._notified_release_tag: str | None = None
        self._register_handlers()

    def authorized(self, user_id: int) -> bool:
        return user_id == self.settings.tg_user_id

    def command(self, name: str, user_id: int) -> str:
        if not self.authorized(user_id):
            return PRIVATE_REPLY
        if name == "start":
            return "Personal HH assistant is ready. Use /status to inspect it."
        if name == "pause":
            self.control.paused = True
            self.control.next_run_at = None
            logger.info("agent_paused")
            return "Agent paused."
        if name == "resume":
            self.control.paused = False
            self.control.circuit_reason = ""
            self.control.consecutive_search_errors = 0
            self.control.next_run_at = None
            self.control.wake_event.set()
            logger.info("agent_resumed")
            return "Agent resumed."
        if name == "status":
            stats = self.database.stats()
            processed = sum(stats.values())
            version_str = f"v{__version__}"
            auto_apply = self.settings.auto_apply
            auto_line = (
                "enabled "
                f"(≥{auto_apply.min_confidence:.2f}, "
                f"{auto_apply.min_batch_size}–{auto_apply.max_batch_size} за пачку, "
                f"{auto_apply.start_hour:02d}:00–{auto_apply.end_hour:02d}:00 "
                f"{auto_apply.timezone})"
                if auto_apply.enabled
                else "disabled"
            )
            if self.latest_release is not None:
                version_str += f" (доступно обновление {self.latest_release.tag_name})"
            return (
                f"версия: {version_str}\n"
                f"mode: {self.settings.app_mode}\n"
                f"state: {'paused' if self.control.paused else 'running'}\n"
                f"auto apply: {auto_line}\n"
                f"processed: {processed}\n"
                f"applied today: {self.database.applied_today(self.now_factory())}"
            )
        if name == "pending":
            pending = self.database.pending()
            if not pending:
                return "No pending vacancies."
            listing = "\n".join(f"{item.id}: {item.title}" for item in pending)
            url = self.queue_page.url() if self.queue_page is not None else ""
            return f"{listing}\n\n{url}" if url else listing
        if name == "stats":
            lines = [
                f"{status}: {count}"
                for status, count in self.database.stats().items()
                if count
            ]
            if not lines:
                return "No vacancies recorded."
            lines.append(
                f"AI-отклики: {self.database.applied_ai_total()} "
                f"(сегодня {self.database.applied_ai_today(self.now_factory())})"
            )
            return "\n".join(lines)
        if name == "diagnostics":
            return self._diagnostics()
        if name == "cancel":
            if self._captcha_future and not self._captcha_future.done():
                self._captcha_future.set_result(None)
            if self._edit_future and not self._edit_future.done():
                self._edit_future.set_result(None)
            return "Current input request cancelled."
        return "Unknown command."

    def _diagnostics(self) -> str:
        run = self.database.latest_search_run()
        version_str = f"v{__version__}"
        if self.latest_release is not None:
            version_str += f" (доступно: {self.latest_release.tag_name})"
        if run is None:
            return f"версия: {version_str}\nNo search diagnostics recorded."
        started = datetime.fromisoformat(run.started_at)
        finished = datetime.fromisoformat(run.finished_at)
        reasons = ", ".join(
            f"{reason}={count}"
            for reason, count in sorted(run.rejection_reasons.items())
        ) or "нет"
        errors = ", ".join(
            f"{reason}={count}"
            for reason, count in sorted(run.error_reasons.items())
        ) or "нет"
        breaker = (
            f"open ({self.control.circuit_reason})"
            if self.control.circuit_reason
            else "closed"
        )
        if self.control.paused:
            next_run = "paused"
        elif self.control.next_run_at is None:
            next_run = "после текущего цикла"
        else:
            next_run = self._in_schedule_zone(
                self.control.next_run_at
            ).strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"версия: {version_str}\n"
            f"последний цикл: {started.astimezone():%Y-%m-%d %H:%M:%S}\n"
            f"состояние: {run.state}\n"
            f"длительность: {int((finished - started).total_seconds())} с\n"
            f"запросы: {run.query_count}\n"
            f"найдено: {run.found_results}\n"
            f"новые: {run.new_vacancies}\n"
            f"дубли: {run.duplicates}\n"
            f"отклонено — фильтр: {run.rejected_by_filter}, LLM: {run.rejected_by_llm}\n"
            f"причины: {reasons}\n"
            f"карточки: {run.telegram_cards}\n"
            f"ошибки: {run.error_count} ({errors})\n"
            f"последняя ошибка: {run.last_safe_error or 'нет'}\n"
            f"breaker: {breaker}\n"
            f"last breaker: {run.circuit_reason or 'none'}\n"
            f"следующий запуск: {next_run}"
        )

    def _get_mode_name(self) -> str:
        if self.settings.app_mode == "auto":
            return "auto"
        if getattr(self.settings, "auto_apply", None) and self.settings.auto_apply.enabled:
            return "auto"
        return self.settings.app_mode

    def _get_mode_label(self) -> str:
        mode = self._get_mode_name()
        if mode == "auto":
            return "🤖 Автоотклик (без AI/LLM; AI/LLM — вручную)"
        elif mode == "approval":
            return "✋ Полуавтомат (Approval)"
        else:
            return "🧪 Тестовый (Dry Run)"

    def _menu_text_and_markup(self) -> tuple[str, InlineKeyboardMarkup]:
        mode_label = self._get_mode_label()
        state_icon = "⏸ <b>На паузе</b>" if self.control.paused else "▶️ <b>Активен (поиск)</b>"
        stats = self.database.stats()
        processed = sum(stats.values())
        applied = self.database.applied_today(self.now_factory())
        pending_list = self.database.pending()
        pending_count = len(pending_list)
        manual_off = self.database.manual_review_disabled()
        manual_label = (
            "выключена, всё откликается автоматом"
            if manual_off
            else "включена, AI-вакансии приходят вам"
        )

        next_run_str = "на паузе" if self.control.paused else (
            self._in_schedule_zone(self.control.next_run_at).strftime("%H:%M")
            + f" ({self.settings.auto_apply.timezone})"
            if self.control.next_run_at else "скоро / выполняется"
        )

        text = (
            f"🐾 <b>Управление HH AI Agent</b>\n\n"
            f"<b>Режим:</b> {mode_label}\n"
            f"<b>Статус:</b> {state_icon}\n"
            f"<b>Следующий цикл:</b> {next_run_str}\n"
            f"<b>Откликов сегодня:</b> {applied} / {self.settings.max_applications_per_day}\n"
            f"<b>Ожидают решения:</b> {pending_count}\n"
            f"<b>Ручная работа:</b> {manual_label}\n"
            f"<b>Всего обработано:</b> {processed}\n\n"
            f"<i>Выберите действие:</i>"
        )

        pause_btn = (
            InlineKeyboardButton(text="▶️ Запустить", callback_data="ctrl:resume")
            if self.control.paused
            else InlineKeyboardButton(text="⏸ Пауза", callback_data="ctrl:pause")
        )
        search_btn = InlineKeyboardButton(text="🔄 Поиск сейчас", callback_data="ctrl:search")
        mode_btn = InlineKeyboardButton(text=f"⚙️ Режим ({self._get_mode_name()})", callback_data="ctrl:mode_menu")
        stats_btn = InlineKeyboardButton(text="📊 Статистика", callback_data="ctrl:stats")
        pending_btn = InlineKeyboardButton(text=f"⏳ Очередь ({pending_count})", callback_data="ctrl:pending")
        bump_btn = InlineKeyboardButton(text="🚀 Поднять резюме", callback_data="ctrl:bump")
        manual_btn = InlineKeyboardButton(
            text=f"🧑‍💻 Ручная работа: {'выкл' if manual_off else 'вкл'}",
            callback_data="ctrl:toggle_manual",
        )
        refresh_btn = InlineKeyboardButton(text="🔄 Обновить статус", callback_data="ctrl:menu")

        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [pause_btn, search_btn],
                [mode_btn],
                [stats_btn, pending_btn],
                [manual_btn],
                [bump_btn, refresh_btn],
            ]
        )
        return text, keyboard

    def _mode_menu_text_and_markup(self) -> tuple[str, InlineKeyboardMarkup]:
        current = self._get_mode_name()

        text = (
            "⚙️ <b>Выберите режим работы:</b>\n\n"
            "🧪 <b>Dry Run (Тест):</b> поиск и скоринг вакансий, подготовка сопроводительных писем в Telegram без кнопок и без реальных откликов.\n\n"
            "✋ <b>Approval (Полуавтомат):</b> бот находит вакансии, пишет письма и присылает карточки с кнопками <code>[Откликнуться]</code> и <code>[Пропустить]</code>. Отклик отправляется только после твоего подтверждения.\n\n"
            "🤖 <b>Auto-Apply (Автомат):</b> бот сам откликается на подходящие вакансии, заполняет анкеты по легенде и решает капчу. Если в меню выключена «Ручная работа», AI/LLM-вакансии тоже откликаются автоматом; если включена — присылаются вам карточкой. Формы с файлами и тестовыми приходят в Telegram в любом случае."
        )

        dry_mark = "✅ " if current == "dry_run" else ""
        appr_mark = "✅ " if current == "approval" else ""
        auto_mark = "✅ " if current == "auto" else ""

        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text=f"{dry_mark}🧪 Dry Run (Тест)", callback_data="mode:dry_run")],
                [InlineKeyboardButton(text=f"{appr_mark}✋ Approval (Полуавтомат)", callback_data="mode:approval")],
                [InlineKeyboardButton(text=f"{auto_mark}🤖 Auto-Apply (Автомат)", callback_data="mode:auto")],
                [InlineKeyboardButton(text="⬅️ Назад в меню", callback_data="ctrl:menu")],
            ]
        )
        return text, keyboard

    def _set_mode(self, mode: str) -> tuple[str, bool]:
        mode = mode.lower().strip()
        if mode in {"dry_run", "dry", "test", "тест"}:
            object.__setattr__(self.settings, "app_mode", "dry_run")
            object.__setattr__(self.settings, "enable_real_apply", False)
            if getattr(self.settings, "auto_apply", None):
                new_auto = replace(self.settings.auto_apply, enabled=False)
                object.__setattr__(self.settings, "auto_apply", new_auto)
            self._persist_env_settings({
                "APP_MODE": "dry_run",
                "ENABLE_REAL_APPLY": "false",
                "AUTO_APPLY_ENABLED": "false",
            })
            return "🧪 Dry Run (Тестовый режим)", True
        elif mode in {"approval", "approve", "полуавтомат", "апрув"}:
            object.__setattr__(self.settings, "app_mode", "approval")
            object.__setattr__(self.settings, "enable_real_apply", True)
            if getattr(self.settings, "auto_apply", None):
                new_auto = replace(self.settings.auto_apply, enabled=False)
                object.__setattr__(self.settings, "auto_apply", new_auto)
            self._persist_env_settings({
                "APP_MODE": "approval",
                "ENABLE_REAL_APPLY": "true",
                "AUTO_APPLY_ENABLED": "false",
            })
            return "✋ Approval (Полуавтомат — подтверждение кнопкой)", True
        elif mode in {"auto", "auto_apply", "авто", "автомат"}:
            object.__setattr__(self.settings, "app_mode", "auto")
            object.__setattr__(self.settings, "enable_real_apply", True)
            if getattr(self.settings, "auto_apply", None):
                new_auto = replace(self.settings.auto_apply, enabled=True)
                object.__setattr__(self.settings, "auto_apply", new_auto)
            self._persist_env_settings({
                "APP_MODE": "auto",
                "ENABLE_REAL_APPLY": "true",
                "AUTO_APPLY_ENABLED": "true",
            })
            return "🤖 Auto-Apply (Автоотклик, AI/LLM — в Telegram)", True
        return f"Неизвестный режим: {mode}", False

    def _persist_env_settings(self, updates: dict[str, str]) -> None:
        for env_path in (Path("/app/.env"), Path(".env")):
            if env_path.exists():
                try:
                    text_env = env_path.read_text(encoding="utf-8")
                    for key, value in updates.items():
                        pattern = rf"^{re.escape(key)}=.*$"
                        new_line = f'{key}="{value}"' if value.lower() not in {"true", "false"} else f"{key}={value}"
                        if re.search(pattern, text_env, flags=re.MULTILINE):
                            text_env = re.sub(pattern, new_line, text_env, flags=re.MULTILINE)
                        else:
                            text_env += f"\n{new_line}"
                    env_path.write_text(text_env, encoding="utf-8")
                    logger.info("persisted_env_settings path=%s updates=%s", env_path, updates)
                    break
                except Exception as exc:
                    logger.warning("persist_env_failed path=%s error=%s", env_path, exc)

    def _register_handlers(self) -> None:
        for name in (
            "start",
            "menu",
            "help",
            "status",
            "pause",
            "stop",
            "resume",
            "run",
            "mode",
            "search",
            "pending",
            "bump",
            "raise",
            "stats",
            "diagnostics",
            "mistral_keys",
            "keys",
            "cancel",
        ):
            self.dispatcher.message.register(self._command_handler, Command(name))
        self.dispatcher.callback_query.register(
            self._callback_handler,
            F.data.startswith("apply:")
            | F.data.startswith("manual:")
            | F.data.startswith("copy:")
            | F.data.startswith("skip:")
            | F.data.startswith("edit:")
            | F.data.startswith("mk:")
            | F.data.startswith("ctrl:")
            | F.data.startswith("mode:")
            | F.data.startswith("noop:"),
        )
        self.dispatcher.message.register(self._text_handler)

    async def _command_handler(self, message: Message) -> None:
        parts = (message.text or "").split(maxsplit=1)
        name = parts[0].lstrip("/").split("@")[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        if not self.authorized(message.from_user.id):
            await message.answer(PRIVATE_REPLY)
            return
        if name in {"mistral_keys", "keys"}:
            await self._send_mistral_key_menu(message)
            return
        if name == "cancel" and self._mistral_key_input is not None:
            self._mistral_key_input = None
            await message.answer("Current Mistral key input cancelled.")
            return

        if name in {"bump", "raise", "поднять"}:
            if not self.hh_client:
                await message.answer("Функция поднятия резюме недоступна.")
                return
            from zoneinfo import ZoneInfo
            msk_now = datetime.now(ZoneInfo("Europe/Moscow"))
            await message.answer("⏳ Проверяю резюме на hh.ru...")
            bump_result = await self.hh_client.bump_resume()
            success, msg = bump_result.ok, bump_result.message
            self.database.record_resume_bump(
                self.now_factory(),
                self.settings.profile.hh.resume_name,
                success,
                msg,
            )
            time_str = msk_now.strftime("%H:%M")
            if success:
                res_text = (
                    f"🚀 <b>Резюме успешно поднято в поиске!</b>\n\n"
                    f"📄 Резюме: <b>{html.escape(self.settings.profile.hh.resume_name)}</b>\n"
                    f"🕒 Время (МСК): {time_str}"
                )
            else:
                res_text = f"ℹ️ <b>Поднятие резюме:</b>\n{html.escape(msg)}\n(МСК: {time_str})"
            await message.answer(res_text, parse_mode="HTML")
            return

        if name in {"start", "menu", "help"}:
            text, markup = self._menu_text_and_markup()
            await message.answer(text, reply_markup=markup, parse_mode="HTML")
            return

        if name in {"status", "статус"}:
            text, markup = self._menu_text_and_markup()
            await message.answer(text, reply_markup=markup, parse_mode="HTML")
            return

        if name in {"mode", "режим"}:
            if args:
                label, ok = self._set_mode(args)
                await message.answer(f"✅ Режим изменен на:\n<b>{label}</b>" if ok else f"❌ {label}", parse_mode="HTML")
            else:
                text, markup = self._mode_menu_text_and_markup()
                await message.answer(text, reply_markup=markup, parse_mode="HTML")
            return

        if name in {"pause", "stop", "пауза", "стоп"}:
            self.control.paused = True
            self.control.next_run_at = None
            logger.info("agent_paused")
            text, markup = self._menu_text_and_markup()
            await message.answer("⏸ <b>Бот поставлен на паузу.</b>\nПоиск вакансий приостановлен.", reply_markup=markup, parse_mode="HTML")
            return

        if name in {"resume", "run", "старт", "пуск"}:
            self.control.paused = False
            self.control.circuit_reason = ""
            self.control.consecutive_search_errors = 0
            self.control.next_run_at = None
            self.control.wake_event.set()
            logger.info("agent_resumed")
            text, markup = self._menu_text_and_markup()
            await message.answer("▶️ <b>Бот запущен!</b>\nПоиск вакансий активен.", reply_markup=markup, parse_mode="HTML")
            return

        if name in {"search", "поиск"}:
            if self.control.paused:
                await message.answer("⚠️ Бот на паузе. Сначала запустите его кнопкой «Запустить» или командой /resume.")
            else:
                self.control.next_run_at = None
                self.control.force_run = True
                self.control.wake_event.set()
                await message.answer("🔄 <b>Поиск запущен прямо сейчас!</b>", parse_mode="HTML")
            return

        await message.answer(self.command(name, message.from_user.id))

    async def _callback_handler(self, callback: CallbackQuery) -> None:
        if not self.authorized(callback.from_user.id):
            await self._answer_callback(callback, PRIVATE_REPLY, show_alert=True)
            return
        action, separator, job_id = (callback.data or "").partition(":")
        if len((callback.data or "").encode()) > 64:
            await self._answer_callback(
                callback, "Некорректное действие.", show_alert=True
            )
            return
        if action == "mk":
            await self._mistral_key_callback(callback, (callback.data or "").split(":"))
            return
        if not separator or action not in {"apply", "manual", "skip", "edit", "copy", "ctrl", "mode", "noop"} or not job_id:
            await self._answer_callback(
                callback, "Некорректное действие.", show_alert=True
            )
            return
        if action == "ctrl":
            ctrl_act = job_id
            if ctrl_act == "menu":
                text, markup = self._menu_text_and_markup()
                if callback.message:
                    try:
                        await callback.message.edit_text(text, reply_markup=markup, parse_mode="HTML")
                    except Exception:
                        pass
                await self._answer_callback(callback, "Меню обновлено")
                return

            elif ctrl_act == "mode_menu":
                text, markup = self._mode_menu_text_and_markup()
                if callback.message:
                    try:
                        await callback.message.edit_text(text, reply_markup=markup, parse_mode="HTML")
                    except Exception:
                        pass
                await self._answer_callback(callback, "Выбор режима")
                return

            elif ctrl_act == "pause":
                self.control.paused = True
                self.control.next_run_at = None
                logger.info("agent_paused_via_callback")
                text, markup = self._menu_text_and_markup()
                if callback.message:
                    try:
                        await callback.message.edit_text(text, reply_markup=markup, parse_mode="HTML")
                    except Exception:
                        pass
                await self._answer_callback(callback, "⏸ Бот на паузе")
                return

            elif ctrl_act == "resume":
                self.control.paused = False
                self.control.circuit_reason = ""
                self.control.consecutive_search_errors = 0
                self.control.next_run_at = None
                self.control.wake_event.set()
                logger.info("agent_resumed_via_callback")
                text, markup = self._menu_text_and_markup()
                if callback.message:
                    try:
                        await callback.message.edit_text(text, reply_markup=markup, parse_mode="HTML")
                    except Exception:
                        pass
                await self._answer_callback(callback, "▶️ Бот запущен")
                return

            elif ctrl_act == "search":
                slots = self.database.available_application_slots(
                    daily_limit=self.settings.max_applications_per_day,
                    now=self.now_factory(),
                )
                if self.control.paused:
                    await self._answer_callback(callback, "⚠️ Сначала запустите бота!", show_alert=True)
                elif slots <= 0:
                    await self._answer_callback(
                        callback,
                        f"⚠️ Дневной лимит откликов исчерпан "
                        f"({self.database.applied_today(self.now_factory())}"
                        f" / {self.settings.max_applications_per_day}).\n"
                        "Новые отклики начнутся после полуночи.",
                        show_alert=True,
                    )
                else:
                    self.control.next_run_at = None
                    self.control.force_run = True
                    self.control.wake_event.set()
                    await self._answer_callback(callback, "🔄 Поиск запущен!", show_alert=True)
                    text, markup = self._menu_text_and_markup()
                    if callback.message:
                        try:
                            await callback.message.edit_text(text, reply_markup=markup, parse_mode="HTML")
                        except Exception:
                            pass
                return

            elif ctrl_act == "bump":
                if not self.hh_client:
                    await self._answer_callback(callback, "Функция поднятия резюме недоступна", show_alert=True)
                    return
                from zoneinfo import ZoneInfo
                msk_now = datetime.now(ZoneInfo("Europe/Moscow"))
                await self._answer_callback(callback, "⏳ Проверяю резюме на hh.ru...", show_alert=False)
                bump_result = await self.hh_client.bump_resume()
                success, msg = bump_result.ok, bump_result.message
                self.database.record_resume_bump(
                    self.now_factory(),
                    self.settings.profile.hh.resume_name,
                    success,
                    msg,
                )
                time_str = msk_now.strftime("%H:%M")
                if success:
                    alert_text = (
                        f"🚀 <b>Резюме успешно поднято в поиске!</b>\n\n"
                        f"📄 Резюме: <b>{html.escape(self.settings.profile.hh.resume_name)}</b>\n"
                        f"🕒 Время (МСК): {time_str}"
                    )
                else:
                    alert_text = f"ℹ️ <b>Поднятие резюме:</b>\n{html.escape(msg)}\n(МСК: {time_str})"
                try:
                    await self.bot.send_message(
                        chat_id=callback.from_user.id,
                        text=alert_text,
                        parse_mode="HTML",
                    )
                except Exception:
                    pass
                return

            elif ctrl_act == "stats":
                stats_str = self.command("stats", callback.from_user.id)
                await self._answer_callback(callback, f"📊 Статистика:\n{stats_str}", show_alert=True)
                return

            elif ctrl_act == "pending":
                await self._send_queue_index(callback)
                return

            elif ctrl_act == "toggle_manual":
                new_state = not self.database.manual_review_disabled()
                self.database.set_manual_review_disabled(new_state)
                logger.info("manual_review_toggled disabled=%s", new_state)
                text, markup = self._menu_text_and_markup()
                if callback.message:
                    try:
                        await callback.message.edit_text(text, reply_markup=markup, parse_mode="HTML")
                    except Exception:
                        pass
                await self._answer_callback(
                    callback,
                    "🧑‍💻 Всё автоматом: AI-вакансии откликаются без вашего участия"
                    if new_state
                    else "✋ AI-вакансии снова будут приходить вам",
                    show_alert=True,
                )
                return

        if action == "mode":
            target_mode = job_id
            label, ok = self._set_mode(target_mode)
            await self._answer_callback(callback, f"✅ Режим: {label}", show_alert=True)
            text, markup = self._menu_text_and_markup()
            if callback.message:
                try:
                    await callback.message.edit_text(text, reply_markup=markup, parse_mode="HTML")
                except Exception:
                    pass
            return
        if action == "noop":
            await self._answer_callback(callback, "Вакансия уже отмечена ✅")
            return

        if action == "manual":
            result = self.approval_service.manual_apply(job_id, callback.from_user.id)
            if result.ok:
                await self._answer_callback(
                    callback, "✅ Отлично! Ручной отклик на сайте засчитан в статистику."
                )
                from zoneinfo import ZoneInfo
                time_msk = datetime.now(ZoneInfo("Europe/Moscow")).strftime("%d.%m %H:%M")
                vacancy = self.database.get(job_id)
                if callback.message and vacancy:
                    status_line = f"✅ <b>ОТКЛИКНУЛСЯ НА САЙТЕ</b> · <i>{time_msk} МСК</i>\n"
                    new_text = self._html_card(vacancy, status_badge=status_line)
                    new_keyboard = InlineKeyboardMarkup(
                        inline_keyboard=[
                            [
                                InlineKeyboardButton(
                                    text=f"✅ Откликнулся · {time_msk}",
                                    callback_data=f"noop:{vacancy.id}",
                                ),
                                InlineKeyboardButton(
                                    text="📋 Скопировать письмо",
                                    callback_data=f"copy:{vacancy.id}",
                                ),
                            ]
                        ]
                    )
                    try:
                        await callback.message.edit_text(
                            text=new_text,
                            parse_mode="HTML",
                            reply_markup=new_keyboard,
                            disable_web_page_preview=True,
                        )
                    except TelegramBadRequest as exc:
                        if "message is not modified" not in str(exc).lower():
                            logger.warning("edit_card_status_failed job_id=%s error=%s", job_id, exc)
                    except Exception as exc:
                        logger.warning("edit_card_status_failed job_id=%s error=%s", job_id, exc)
            else:
                await self._answer_callback(
                    callback, result.message, show_alert=True
                )
            return

        if action == "copy":
            vacancy = self.database.get(job_id)
            letter = vacancy.cover_letter if vacancy else ""
            if not letter:
                await self._answer_callback(
                    callback, "Письмо не найдено.", show_alert=True
                )
                return
            await self._answer_callback(
                callback, "📋 Письмо отправлено ниже — тапните по нему, чтобы скопировать"
            )
            max_len = 4000
            chunks = [letter[i:i + max_len] for i in range(0, len(letter), max_len)]
            for index, chunk in enumerate(chunks):
                header = (
                    "📋 <b>Сопроводительное письмо</b> "
                    "<i>(тапните по тексту — он скопируется)</i>\n"
                    if index == 0 else ""
                )
                await self.bot.send_message(
                    chat_id=callback.from_user.id,
                    text=f"{header}<pre>{html.escape(chunk)}</pre>",
                    parse_mode="HTML",
                )
            logger.info("cover_letter_copy_sent job_id=%s", job_id)
            return

        if action == "apply":
            vacancy = self.database.get(job_id)
            if not vacancy:
                await self._answer_callback(callback, "Вакансия не найдена", show_alert=True)
                return

            from zoneinfo import ZoneInfo
            time_msk = datetime.now(ZoneInfo("Europe/Moscow")).strftime("%d.%m %H:%M")

            # 1. Если на вакансию уже откликнулись ранее — сразу помечаем карточку
            if vacancy.status.value == "applied":
                await self._answer_callback(callback, "✅ Вы уже откликнулись на эту вакансию!", show_alert=True)
                if callback.message:
                    applied_time = (
                        datetime.fromisoformat(vacancy.applied_at).astimezone(ZoneInfo("Europe/Moscow")).strftime("%d.%m %H:%M")
                        if vacancy.applied_at else time_msk
                    )
                    status_line = f"✅ <b>ОТКЛИК ОТПРАВЛЕН БОТОМ</b> · <i>{applied_time} МСК</i>\n"
                    new_text = self._html_card(vacancy, status_badge=status_line)
                    new_keyboard = InlineKeyboardMarkup(
                        inline_keyboard=[
                            [
                                InlineKeyboardButton(
                                    text=f"✅ Отправлен · {applied_time}",
                                    callback_data=f"noop:{vacancy.id}",
                                ),
                                InlineKeyboardButton(
                                    text="📋 Скопировать письмо",
                                    callback_data=f"copy:{vacancy.id}",
                                ),
                            ]
                        ]
                    )
                    try:
                        await callback.message.edit_text(
                            text=new_text,
                            parse_mode="HTML",
                            reply_markup=new_keyboard,
                            disable_web_page_preview=True,
                        )
                    except Exception as exc:
                        logger.warning("edit_card_status_failed job_id=%s error=%s", job_id, exc)
                return

            # 2. Если прямо сейчас уже идёт отклик через браузер
            if vacancy.status.value == "applying":
                await self._answer_callback(callback, "⏳ Отклик уже отправляется через браузер прямо сейчас...", show_alert=True)
                return

            # 3. МГНОВЕННЫЙ визуальный отклик на карточке:
            # Сразу меняем статус и кнопки на карточке, чтобы пользователь видел результат нажатия
            if callback.message:
                in_progress_badge = "⏳ <b>ОТПРАВЛЯЕМ ОТКЛИК ЧЕРЕЗ БРАУЗЕР...</b>\n"
                progress_text = self._html_card(vacancy, status_badge=in_progress_badge)
                progress_keyboard = InlineKeyboardMarkup(
                    inline_keyboard=[
                        [
                            InlineKeyboardButton(
                                text="⏳ Отправка отклика...",
                                callback_data=f"noop:{vacancy.id}",
                            ),
                            InlineKeyboardButton(
                                text="📋 Скопировать письмо",
                                callback_data=f"copy:{vacancy.id}",
                            ),
                        ]
                    ]
                )
                try:
                    await callback.message.edit_text(
                        text=progress_text,
                        parse_mode="HTML",
                        reply_markup=progress_keyboard,
                        disable_web_page_preview=True,
                    )
                except Exception as exc:
                    logger.warning("edit_card_progress_failed job_id=%s error=%s", job_id, exc)

            await callback.answer("⏳ Отправляем отклик через браузер...", show_alert=False)

            result = await self.approval_service.approve_and_apply(
                job_id, callback.from_user.id
            )
            vacancy = self.database.get(job_id)
            if result.ok:
                await self.notify(f"✓ Отклик отправлен: {vacancy.title if vacancy else ''}")
                if callback.message and vacancy:
                    status_line = f"✅ <b>ОТКЛИК ОТПРАВЛЕН БОТОМ</b> · <i>{time_msk} МСК</i>\n"
                    new_text = self._html_card(vacancy, status_badge=status_line)
                    new_keyboard = InlineKeyboardMarkup(
                        inline_keyboard=[
                            [
                                InlineKeyboardButton(
                                    text=f"✅ Отправлен · {time_msk}",
                                    callback_data=f"noop:{vacancy.id}",
                                ),
                                InlineKeyboardButton(
                                    text="📋 Скопировать письмо",
                                    callback_data=f"copy:{vacancy.id}",
                                ),
                            ]
                        ]
                    )
                    try:
                        await callback.message.edit_text(
                            text=new_text,
                            parse_mode="HTML",
                            reply_markup=new_keyboard,
                            disable_web_page_preview=True,
                        )
                    except Exception as exc:
                        logger.warning("edit_card_status_failed job_id=%s error=%s", job_id, exc)
            else:
                if vacancy and "questionnaire" in vacancy.error_text:
                    await self.notify_questionnaire_required(vacancy.title, vacancy.url)
                    if callback.message:
                        status_line = "📋 <b>ТРЕБУЕТСЯ АНКЕТА / ТЕСТОВОЕ НА САЙТЕ</b>\n"
                        new_text = self._html_card(vacancy, status_badge=status_line)
                        new_keyboard = InlineKeyboardMarkup(
                            inline_keyboard=[
                                [
                                    InlineKeyboardButton(
                                        text="🌐 Я откликнулся на сайте",
                                        callback_data=f"manual:{vacancy.id}",
                                    ),
                                    InlineKeyboardButton(
                                        text="❌ Пропустить",
                                        callback_data=f"skip:{vacancy.id}",
                                    ),
                                ],
                                [
                                    InlineKeyboardButton(
                                        text="📋 Скопировать письмо",
                                        callback_data=f"copy:{vacancy.id}",
                                    ),
                                ],
                            ]
                        )
                        try:
                            await callback.message.edit_text(
                                text=new_text,
                                parse_mode="HTML",
                                reply_markup=new_keyboard,
                                disable_web_page_preview=True,
                            )
                        except Exception as exc:
                            logger.warning(
                                "edit_card_status_failed job_id=%s error=%s", job_id, exc
                            )
                else:
                    await self.notify(f"✗ Не удалось отправить отклик: {result.message}")
                    if callback.message and vacancy:
                        status_line = f"❌ <b>ОШИБКА ОТКЛИКА: {html.escape(result.message)}</b>\n"
                        err_text = self._html_card(vacancy, status_badge=status_line)
                        err_keyboard = InlineKeyboardMarkup(
                            inline_keyboard=[
                                [
                                    InlineKeyboardButton(
                                        text="🔄 Повторить",
                                        callback_data=f"apply:{vacancy.id}",
                                    ),
                                    InlineKeyboardButton(
                                        text="🌐 Я откликнулся на сайте",
                                        callback_data=f"manual:{vacancy.id}",
                                    ),
                                ],
                                [
                                    InlineKeyboardButton(
                                        text="❌ Пропустить",
                                        callback_data=f"skip:{vacancy.id}",
                                    ),
                                    InlineKeyboardButton(
                                        text="📋 Скопировать письмо",
                                        callback_data=f"copy:{vacancy.id}",
                                    ),
                                ],
                            ]
                        )
                        try:
                            await callback.message.edit_text(
                                text=err_text,
                                parse_mode="HTML",
                                reply_markup=err_keyboard,
                                disable_web_page_preview=True,
                            )
                        except Exception:
                            pass
        elif action == "edit":
            self._mistral_key_input = None
            await callback.answer("Пришлите новый текст сопроводительного письма сообщением (или /cancel):", show_alert=True)
            loop = asyncio.get_running_loop()
            self._edit_job_id = job_id
            self._edit_future = loop.create_future()
            try:
                new_letter = await asyncio.wait_for(self._edit_future, timeout=300)
                if new_letter:
                    if not has_required_portfolio(
                        new_letter, self.settings.profile.cover_letter.required_portfolio_url
                    ):
                        await self.notify("Письмо не сохранено: добавьте обязательную ссылку на портфолио.")
                        return
                    self.database.update_cover_letter(job_id, new_letter)
                    await self.notify("✓ Сопроводительное письмо обновлено!")
                    updated_vacancy = self.database.get(job_id)
                    if updated_vacancy:
                        await self.send_preview(updated_vacancy, include_actions=True)
                else:
                    await self.notify("Редактирование письма отменено.")
            except TimeoutError:
                await self.notify("Время ожидания редактирования письма истекло.")
            finally:
                self._edit_job_id = None
                self._edit_future = None
        else:
            # skip выполняется быстро — можно отвечать обычным способом.
            result = self.approval_service.skip(job_id, callback.from_user.id)
            await self._answer_callback(
                callback, result.message, show_alert=not result.ok
            )
            if result.ok and callback.message:
                try:
                    await callback.message.delete()
                except Exception:
                    pass

    async def _send_mistral_key_menu(self, message: Any) -> None:
        if self.mistral_keys is None:
            await message.answer("Управление ключами Mistral недоступно.")
            return
        try:
            keys = self.mistral_keys.list_keys()
        except LLMError as exc:
            await message.answer(f"Ключи Mistral недоступны: {exc.category}.")
            return
        text, keyboard = self._mistral_key_menu(keys)
        await message.answer(text, reply_markup=keyboard)

    @classmethod
    def _mistral_key_menu(
        cls, keys: tuple[MistralKeyView, ...]
    ) -> tuple[str, InlineKeyboardMarkup | None]:
        lines = ["Ключи Mistral:"]
        rows: list[list[InlineKeyboardButton]] = []
        for key in keys:
            marker = (
                "✅ текущий"
                if key.is_current
                else {
                    "ready": "✅ рабочий",
                    "cooldown": f"⏸ до {key.cooldown_until.astimezone():%H:%M}",
                    "disabled": "❌ отключён",
                }[key.status]
            )
            lines.append(f"{key.id}. {marker} ····{key.suffix}")
            rows.append(
                [
                    InlineKeyboardButton(
                        text=f"Проверить {key.id}",
                        callback_data=f"mk:check:{key.id}",
                    ),
                    InlineKeyboardButton(
                        text=f"Удалить {key.id}",
                        callback_data=f"mk:delete:{key.id}",
                    ),
                ]
            )
        if not keys:
            lines.append("Ключей нет.")
        rows.append(
            [
                InlineKeyboardButton(text="➕ Добавить", callback_data="mk:add"),
                InlineKeyboardButton(text="Проверить все", callback_data="mk:check"),
            ]
        )
        rows.append(
            [InlineKeyboardButton(text="Обновить", callback_data="mk:list")]
        )
        return "\n".join(lines), cls._mistral_keyboard(rows)

    async def _mistral_key_callback(
        self, callback: CallbackQuery, parts: list[str]
    ) -> None:
        if self.mistral_keys is None:
            await self._answer_callback(
                callback, "Управление ключами Mistral недоступно.", show_alert=True
            )
            return
        if parts == ["mk", "add"]:
            if self._captcha_future and not self._captcha_future.done():
                await self._answer_callback(
                    callback, "Сначала завершите CAPTCHA.", show_alert=True
                )
                return
            if self._edit_future is not None:
                await self._answer_callback(
                    callback,
                    "Сначала завершите редактирование письма.",
                    show_alert=True,
                )
                return
            self._mistral_key_input = MistralKeyInputSession(
                self.now_factory() + MISTRAL_KEY_INPUT_TTL
            )
            await self._answer_callback(
                callback, "Отправьте ключ одним сообщением."
            )
            return
        if parts == ["mk", "list"]:
            try:
                text, keyboard = self._mistral_key_menu(self.mistral_keys.list_keys())
            except LLMError as exc:
                await self._answer_callback(
                    callback, f"Ключи Mistral недоступны: {exc.category}.", show_alert=True
                )
                return
            await self._edit_mistral_message(callback, text, reply_markup=keyboard)
            return
        if parts == ["mk", "check"]:
            await self._answer_callback(callback, "Проверяем ключи.")
            try:
                results = await self.mistral_keys.check_all()
            except LLMError as exc:
                await self._edit_mistral_message(
                    callback,
                    f"Проверка не выполнена: {exc.category}.",
                    acknowledge=False,
                )
                return
            text = "\n".join(self._mistral_check_text(result) for result in results)
            await self._edit_mistral_message(
                callback, text or "Ключей для проверки нет.", acknowledge=False
            )
            return
        if len(parts) != 3 or not parts[2].isdecimal():
            await self._answer_callback(
                callback, "Некорректное действие.", show_alert=True
            )
            return
        action, key_id = parts[1], int(parts[2])
        if action == "delete":
            keyboard = self._mistral_keyboard(
                [
                    [
                        InlineKeyboardButton(
                            text="Удалить", callback_data=f"mk:confirm:{key_id}"
                        ),
                        InlineKeyboardButton(text="Отмена", callback_data="mk:list"),
                    ]
                ]
            )
            await self._edit_mistral_message(
                callback, f"Удалить ключ {key_id}?", reply_markup=keyboard
            )
            return
        if action == "check":
            await self._answer_callback(callback, "Проверяем ключ.")
            try:
                result = await self.mistral_keys.check_key(key_id)
            except LLMError as exc:
                await self._edit_mistral_message(
                    callback,
                    f"Проверка не выполнена: {exc.category}.",
                    acknowledge=False,
                )
                return
            if result is None:
                await self._edit_mistral_message(
                    callback, "Ключ не найден.", acknowledge=False
                )
                return
            await self._edit_mistral_message(
                callback, self._mistral_check_text(result), acknowledge=False
            )
            return
        if action == "confirm":
            try:
                deleted = await self.mistral_keys.delete_key(key_id)
            except LLMError as exc:
                await self._answer_callback(
                    callback, f"Ключ не удалён: {exc.category}.", show_alert=True
                )
                return
            if not deleted:
                await self._answer_callback(
                    callback, "Ключ не найден.", show_alert=True
                )
                return
            await self._edit_mistral_message(callback, f"Ключ {key_id} удалён.")
            return
        await self._answer_callback(
            callback, "Некорректное действие.", show_alert=True
        )

    @staticmethod
    def _mistral_check_text(result: MistralKeyCheckResult) -> str:
        outcome = "работает" if result.success else result.error_type
        return f"Ключ {result.key.id} ····{result.key.suffix}: {outcome}."

    @staticmethod
    def _mistral_keyboard(
        rows: list[list[InlineKeyboardButton]],
    ) -> InlineKeyboardMarkup | None:
        if any(
            len(button.callback_data.encode()) > 64
            for row in rows
            for button in row
        ):
            return None
        return InlineKeyboardMarkup(inline_keyboard=rows)

    async def _edit_mistral_message(
        self,
        callback: CallbackQuery,
        text: str,
        *,
        reply_markup: InlineKeyboardMarkup | None = None,
        acknowledge: bool = True,
    ) -> None:
        if callback.message is None:
            await self._answer_callback(
                callback, "Меню недоступно.", show_alert=True
            )
            return
        try:
            await callback.message.edit_text(text=text, reply_markup=reply_markup)
        except TelegramAPIError:
            if acknowledge:
                await self._answer_callback(
                    callback,
                    "Действие выполнено, но меню не обновлено.",
                    show_alert=True,
                )
            return
        if acknowledge:
            await self._answer_callback(callback, "Готово.")

    async def _send_queue_index(self, callback: CallbackQuery) -> None:
        """Send a one-hour link to the web page that clears the pending queue."""
        url = self.queue_page.url() if self.queue_page is not None else ""
        pending = self.database.pending()
        if not url:
            pending_str = self.command("pending", callback.from_user.id)
            await self._answer_callback(
                callback, f"⏳ Очередь:\n{pending_str}", show_alert=True
            )
            return
        text = (
            f"⏳ <b>Очередь на отклик: {len(pending)}</b>\n"
            "Откройте список, пройдите по ссылкам и отметьте каждую вакансию.\n"
            "Ссылка действует 1 час."
        )
        markup = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="🌐 Открыть очередь", url=url)]
            ]
        )
        try:
            await self.bot.send_message(
                chat_id=callback.from_user.id,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=True,
                reply_markup=markup,
            )
            await self._answer_callback(callback, "Ссылка отправлена")
        except Exception as exc:
            logger.warning("queue_index_send_failed error=%s", exc)
            await self._answer_callback(
                callback, "Не удалось отправить ссылку", show_alert=True
            )

    @staticmethod
    async def _answer_callback(
        callback: CallbackQuery, text: str, *, show_alert: bool = False
    ) -> None:
        try:
            await callback.answer(text, show_alert=show_alert)
        except TelegramAPIError:
            logger.warning("telegram_callback_answer_failed")

    async def _text_handler(self, message: Message) -> None:
        if not self.authorized(message.from_user.id):
            await message.answer(PRIVATE_REPLY)
            return
        text = message.text
        if text is None or (
            text.startswith("/") and self._mistral_key_input is None
        ):
            return
        if self._captcha_future and not self._captcha_future.done():
            self._captcha_future.set_result(text.strip())
            await message.answer("CAPTCHA input received.")
            return
        if self._edit_future and not self._edit_future.done() and message.text:
            self._edit_future.set_result(message.text.strip())
            return
        lowered = text.strip().lower()
        if lowered in {"меню", "menu"}:
            text_menu, markup = self._menu_text_and_markup()
            await message.answer(text_menu, reply_markup=markup, parse_mode="HTML")
            return
        if lowered in {"пауза", "стоп", "pause", "stop"}:
            self.control.paused = True
            self.control.next_run_at = None
            text_menu, markup = self._menu_text_and_markup()
            await message.answer("⏸ <b>Бот поставлен на паузу.</b>", reply_markup=markup, parse_mode="HTML")
            return
        if lowered in {"старт", "пуск", "запуск", "start", "run", "resume"}:
            self.control.paused = False
            self.control.circuit_reason = ""
            self.control.consecutive_search_errors = 0
            self.control.next_run_at = None
            self.control.wake_event.set()
            text_menu, markup = self._menu_text_and_markup()
            await message.answer("▶️ <b>Бот запущен!</b>", reply_markup=markup, parse_mode="HTML")
            return
        if lowered in {"режим", "mode"}:
            text_mode, markup = self._mode_menu_text_and_markup()
            await message.answer(text_mode, reply_markup=markup, parse_mode="HTML")
            return
        if lowered in {"поиск", "искать", "search"}:
            if self.control.paused:
                await message.answer("⚠️ Бот на паузе. Сначала запустите его.")
            else:
                self.control.next_run_at = None
                self.control.force_run = True
                self.control.wake_event.set()
                await message.answer("🔄 <b>Поиск запущен прямо сейчас!</b>", parse_mode="HTML")
            return
        if lowered in {"статус", "инфо", "status", "info"}:
            text_menu, markup = self._menu_text_and_markup()
            await message.answer(text_menu, reply_markup=markup, parse_mode="HTML")
            return

        key_session = self._mistral_key_input
        if key_session is None:
            return
        if self.now_factory() >= key_session.expires_at:
            self._mistral_key_input = None
            await message.answer("Время ввода ключа истекло.")
            return
        raw_key = text.strip()
        try:
            await message.delete()
        except TelegramAPIError:
            self._mistral_key_input = None
            await message.answer(
                "Не удалось удалить сообщение. Удалите его вручную и повторите добавление."
            )
            return
        self._mistral_key_input = None
        try:
            result = await self.mistral_keys.add_key(raw_key)
        except LLMError as exc:
            await message.answer(f"Ключ не добавлен: {exc.category}.")
            return
        await message.answer(f"Ключ {result.id} ····{result.suffix} добавлен.")

    async def send_preview(
        self, vacancy: Vacancy, include_actions: bool, status_badge: str = ""
    ) -> None:
        keyboard = None
        is_ai = self._is_ai_priority(vacancy)
        edit_text = "🔥 ✏️ Написать письмо руками (Приоритет!)" if is_ai else "✏️ Изменить письмо"
        if include_actions:
            keyboard = InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="✅ Откликнуться ботом", callback_data=f"apply:{vacancy.id}"
                        ),
                        InlineKeyboardButton(
                            text="🌐 Я откликнулся на сайте", callback_data=f"manual:{vacancy.id}"
                        ),
                    ],
                    [
                        InlineKeyboardButton(
                            text="❌ Пропустить", callback_data=f"skip:{vacancy.id}"
                        ),
                        InlineKeyboardButton(
                            text=edit_text, callback_data=f"edit:{vacancy.id}"
                        ),
                    ],
                    [
                        InlineKeyboardButton(
                            text="📋 Скопировать письмо", callback_data=f"copy:{vacancy.id}"
                        ),
                    ],
                ]
            )
        else:
            keyboard = InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="🌐 Я откликнулся на сайте", callback_data=f"manual:{vacancy.id}"
                        ),
                        InlineKeyboardButton(
                            text="❌ Пропустить", callback_data=f"skip:{vacancy.id}"
                        ),
                    ],
                    [
                        InlineKeyboardButton(
                            text="📋 Скопировать письмо", callback_data=f"copy:{vacancy.id}"
                        ),
                    ],
                ]
            )
        chat_id = str(self.settings.tg_user_id)
        card = self.database.claim_telegram_card(
            vacancy.id, chat_id, now=datetime.now(UTC)
        )
        if card is None:
            logger.info("telegram_card_send_already_in_progress job_id=%s", vacancy.id)
            return
        text = self._html_card(vacancy, status_badge=status_badge)
        try:
            if card.message_id is not None:
                try:
                    await self.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=card.message_id,
                        text=text,
                        parse_mode="HTML",
                        reply_markup=keyboard,
                        disable_web_page_preview=True,
                    )
                    self.database.complete_telegram_card(
                        vacancy.id, chat_id, card.message_id
                    )
                    logger.info("telegram_card_updated job_id=%s", vacancy.id)
                    return
                except TelegramBadRequest as exc:
                    error = str(exc).lower()
                    if "message is not modified" in error:
                        self.database.complete_telegram_card(
                            vacancy.id, chat_id, card.message_id
                        )
                        return
                    if (
                        "message to edit not found" not in error
                        and "message not found" not in error
                    ):
                        raise
                    self.database.clear_telegram_card_message(vacancy.id, chat_id)
            message = await self.bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode="HTML",
                reply_markup=keyboard,
                disable_web_page_preview=True,
            )
            if message is None or message.message_id is None:
                raise RuntimeError("Telegram did not return a message_id")
            self.database.complete_telegram_card(
                vacancy.id, chat_id, message.message_id
            )
            logger.info(
                "%s job_id=%s message_id=%s",
                "approval_requested" if include_actions else "preview_sent",
                vacancy.id,
                message.message_id,
            )
        except BaseException:
            self.database.release_telegram_card(vacancy.id, chat_id)
            raise

    @staticmethod
    def _is_ai_priority(vacancy: Vacancy) -> bool:
        if vacancy.is_ai_priority:
            return True
        return is_ai_llm_vacancy(vacancy.title)

    @staticmethod
    def _confidence(vacancy: Vacancy) -> str:
        return "нет данных" if vacancy.confidence is None else f"{vacancy.confidence:.0%}"

    @staticmethod
    def _rating(vacancy: Vacancy) -> str | None:
        if vacancy.company_rating is None or vacancy.company_reviews_count is None:
            return None
        rating = f"{vacancy.company_rating:.1f}".replace(".", ",")
        return f"★ {rating}/5 · {vacancy.company_reviews_count} отзывов"

    @staticmethod
    def _fit_lines(vacancy: Vacancy) -> list[tuple[str, str]]:
        return [
            (category, value)
            for line in (vacancy.fit_summary or FIT_SUMMARY_FALLBACK).splitlines()
            for category, separator, value in [line.partition(": ")]
            if separator
        ]

    @classmethod
    def _split_fit_lines(cls, vacancy: Vacancy) -> tuple[list[tuple[str, str]], str | None]:
        fit_items = []
        attention = None
        for category, value in cls._fit_lines(vacancy):
            if category in {"Внимание", "Обратить внимание", "Нюансы"}:
                attention = value
            elif category != "🔥 AI / LLM":
                fit_items.append((category, value))
        return fit_items, attention

    @classmethod
    def _rich_card(cls, vacancy: Vacancy) -> InputRichMessage:
        company_blocks = [
            InputRichBlockParagraph(
                text=[RichTextBold(text="Компания: "), vacancy.company or "не указана"]
            )
        ]
        if rating := cls._rating(vacancy):
            company_blocks.append(
                InputRichBlockParagraph(
                    text=[RichTextBold(text="Рейтинг HH: "), rating]
                )
            )
        fit_items, attention = cls._split_fit_lines(vacancy)
        blocks = [
            InputRichBlockSectionHeading(text=vacancy.title, size=2),
            InputRichBlockParagraph(
                text=RichTextUrl(text="Открыть вакансию", url=vacancy.url)
            ),
            InputRichBlockDivider(),
            InputRichBlockDetails(
                summary="Компания", blocks=company_blocks, is_open=True
            ),
            InputRichBlockSectionHeading(text="Почему мне подходит", size=3),
            *[
                InputRichBlockParagraph(
                    text=[RichTextBold(text=f"{category}: "), value]
                )
                for category, value in fit_items
            ],
        ]
        if attention:
            blocks.extend([
                InputRichBlockSectionHeading(text="⚠️ На что обратить внимание", size=3),
                InputRichBlockParagraph(text=attention),
            ])
        blocks.extend([
            InputRichBlockParagraph(
                text=[RichTextBold(text="Уверенность: "), cls._confidence(vacancy)]
            ),
            InputRichBlockDetails(
                summary="Сопроводительное письмо",
                blocks=[InputRichBlockPreformatted(text=vacancy.cover_letter)],
                is_open=False,
            ),
        ])
        return InputRichMessage(skip_entity_detection=True, blocks=blocks)

    @classmethod
    def _html_card(cls, vacancy: Vacancy, status_badge: str = "") -> str:
        confidence = cls._confidence(vacancy)
        stars = ""
        if vacancy.confidence is not None:
            stars = "⭐" * max(1, min(5, round(vacancy.confidence * 5))) + " "

        company = html.escape(vacancy.company or "не указана")
        rating = cls._rating(vacancy)
        rating_str = f" · {rating}" if rating else ""

        fit_items, attention = cls._split_fit_lines(vacancy)
        fit_lines = [
            f"▫️ <b>{html.escape(cat)}:</b> {html.escape(val)}"
            for cat, val in fit_items
        ]
        fit_section = "\n".join(fit_lines) if fit_lines else "▫️ <b>Опыт:</b> системный анализ в продуктовых командах"

        attention_section = ""
        if attention:
            attention_section = f"\n\n⚠️ <b>Обратить внимание:</b>\n<blockquote>{html.escape(attention)}</blockquote>"

        ai_banner = ""
        if cls._is_ai_priority(vacancy):
            ai_banner = (
                "🔥 <b>[ПРИОРИТЕТНАЯ ВАКАНСИЯ: AI / LLM АГЕНТЫ!]</b>\n"
                "⚡️ <i>Рекомендуется индивидуальное письмо вручную</i>\n\n"
            )

        badge = f"{status_badge}\n" if status_badge else ""
        title = html.escape(vacancy.title)
        url = html.escape(vacancy.url, quote=True)

        letter = html.escape(vacancy.cover_letter or "")
        letter_block = (
            f"✉️ <b>Сопроводительное письмо:</b> <i>(тапните для копирования)</i>\n<pre>{letter}</pre>"
            if letter else "<i>(Сопроводительное письмо не сформировано)</i>"
        )

        return (
            f"{badge}"
            f"{ai_banner}"
            f"{stars}<b>{confidence}</b> · <a href=\"{url}\">{title}</a>\n"
            f"🏢 <b>{company}</b>{rating_str}\n\n"
            f"🎯 <b>Почему подходит:</b>\n"
            f"{fit_section}"
            f"{attention_section}\n\n"
            f"{letter_block}"
        )

    def _in_schedule_zone(self, moment: datetime) -> datetime:
        return moment.astimezone(ZoneInfo(self.settings.auto_apply.timezone))

    async def notify(self, text: str) -> None:
        try:
            await self.bot.send_message(chat_id=self.settings.tg_user_id, text=text)
        except Exception as exc:
            logger.warning("telegram_notify_failed error=%s", exc)

    async def notify_screenshot(self, path: Path, caption: str) -> None:
        try:
            await self.bot.send_photo(
                chat_id=self.settings.tg_user_id,
                photo=FSInputFile(path),
                caption=caption[:1024],
            )
        except Exception as exc:
            logger.warning("telegram_screenshot_failed error=%s", exc)

    async def notify_analysis_failed(self, title: str, url: str, error_type: str) -> None:
        text = (
            f"⚠️ <b>Ошибка AI-анализа вакансии</b>\n"
            f"<b>{html.escape(title)}</b>\n"
            f"<a href=\"{html.escape(url, quote=True)}\">Открыть вакансию</a>\n\n"
            f"Причина: <code>{html.escape(error_type)}</code>"
        )
        try:
            await self.bot.send_message(
                chat_id=self.settings.tg_user_id,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        except Exception as exc:
            logger.warning("notify_analysis_failed_error error=%s", exc)

    async def notify_questionnaire_required(self, title: str, url: str) -> None:
        text = (
            f"📋 <b>Требуется ручной отклик (тестовое / анкета)</b>\n"
            f"<b>{html.escape(title)}</b>\n"
            f"<a href=\"{html.escape(url, quote=True)}\">Открыть вакансию на HH.ru</a>\n\n"
            f"Работодатель требует заполнение анкеты или выполнение тестового задания.\n"
            f"Заполните её на сайте, затем нажмите «Я откликнулся на сайте» в карточке вакансии."
        )
        try:
            await self.bot.send_message(
                chat_id=self.settings.tg_user_id,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        except Exception as exc:
            logger.warning("notify_questionnaire_failed error=%s", exc)

    async def request_captcha(
        self, screenshot: Path, title: str, timeout_seconds: int
    ) -> str | None:
        self._mistral_key_input = None
        loop = asyncio.get_running_loop()
        self._captcha_future = loop.create_future()
        await self.bot.send_photo(
            chat_id=self.settings.tg_user_id,
            photo=FSInputFile(screenshot),
            caption=f"CAPTCHA detected for: {title}\nReply with the text or use /cancel.",
        )
        try:
            return await asyncio.wait_for(self._captcha_future, timeout_seconds)
        except TimeoutError:
            return None
        finally:
            self._captcha_future = None

    async def check_updates(self, notify: bool = True) -> ReleaseInfo | None:
        try:
            release = await check_github_release(current_version=__version__)
        except Exception as exc:
            logger.debug("update_check_failed error=%s", exc)
            return None
        if release is not None:
            self.latest_release = release
            if notify and self._notified_release_tag != release.tag_name:
                self._notified_release_tag = release.tag_name
                await self.notify_update_available(release)
        return release

    async def notify_update_available(self, release: ReleaseInfo) -> None:
        text = (
            f"🚀 <b>Доступно обновление HH Agent {html.escape(release.tag_name)}!</b>\n"
            f"Текущая версия: <code>v{__version__}</code>\n\n"
            f"<a href=\"{html.escape(release.html_url, quote=True)}\">Посмотреть список изменений на GitHub</a>"
        )
        try:
            await self.bot.send_message(
                chat_id=self.settings.tg_user_id,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        except Exception as exc:
            logger.warning("notify_update_available_failed error=%s", exc)

    async def start_polling(self) -> None:
        try:
            await self.bot.set_my_commands([
                BotCommand(command="menu", description="📱 Главное меню управления"),
                BotCommand(command="status", description="ℹ️ Текущий статус бота"),
                BotCommand(command="mode", description="⚙️ Выбор режима (Dry Run / Approval / Auto)"),
                BotCommand(command="pause", description="⏸ Приостановить работу"),
                BotCommand(command="resume", description="▶️ Запустить / возобновить работу"),
                BotCommand(command="search", description="🔄 Запустить поиск сейчас"),
                BotCommand(command="stats", description="📊 Статистика откликов"),
                BotCommand(command="pending", description="⏳ Очередь на апрув"),
                BotCommand(command="bump", description="🚀 Поднять резюме в поиске hh.ru"),
            ])
        except Exception as exc:
            logger.debug("set_my_commands_failed error=%s", exc)
        while True:
            try:
                await self.dispatcher.start_polling(self.bot, handle_signals=False)
                break
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("telegram_polling_error error=%s, retrying in 5s", exc)
                await asyncio.sleep(5)

    async def stop(self) -> None:
        session = getattr(self.bot, "session", None)
        if session is not None:
            await session.close()
