from __future__ import annotations

import hashlib
import hmac
import html
import logging
import time
from collections.abc import Sequence

from aiohttp import web

from config import Settings
from database import Database, Vacancy

logger = logging.getLogger(__name__)

TOKEN_TTL_SECONDS = 3600

ATTENTION_CATEGORIES = {"Внимание", "Обратить внимание", "Нюансы"}


def _signature(secret: str, expires: int) -> str:
    return hmac.new(
        secret.encode(), str(expires).encode(), hashlib.sha256
    ).hexdigest()[:32]


def issue_token(secret: str, *, now: float | None = None, ttl: int = TOKEN_TTL_SECONDS) -> str:
    expires = int(now if now is not None else time.time()) + ttl
    return f"{expires}.{_signature(secret, expires)}"


def token_is_valid(secret: str, token: str, *, now: float | None = None) -> bool:
    if not secret or not token:
        return False
    try:
        expires_text, signature = token.split(".", 1)
        expires = int(expires_text)
    except ValueError:
        return False
    if expires < (now if now is not None else time.time()):
        return False
    return hmac.compare_digest(signature, _signature(secret, expires))


def _attention_line(vacancy: Vacancy) -> str:
    """The same "обратить внимание" note the Telegram card shows."""
    for line in (vacancy.fit_summary or "").splitlines():
        category, separator, value = line.partition(": ")
        if separator and category.strip() in ATTENTION_CATEGORIES:
            return value.strip()
    return ""


def _short_date(value: str) -> str:
    parts = value[:10].split("-")
    return f"{parts[2]}.{parts[1]}" if len(parts) == 3 else value[:10]


def _row(vacancy: Vacancy, token: str, position: int) -> str:
    base = f"/q/{html.escape(token)}/done/{html.escape(vacancy.id)}"
    company = html.escape(vacancy.company) if vacancy.company else "Компания не указана"
    attention = _attention_line(vacancy)
    attention_block = (
        f'<p class="attention"><span>Внимание</span>{html.escape(attention)}</p>'
        if attention
        else ""
    )
    has_letter = bool(vacancy.cover_letter.strip())
    letter_id = f"letter-{html.escape(vacancy.id)}"
    letter = (
        f'<span class="letter-source" id="{letter_id}" hidden>'
        f"{html.escape(vacancy.cover_letter)}</span>"
        if has_letter
        else ""
    )
    copy_button = (
        f'<button class="copy" type="button" data-target="{letter_id}"'
        ' title="Скопировать письмо" aria-label="Скопировать письмо">'
        '<svg class="ico" viewBox="0 0 24 24" aria-hidden="true">'
        '<rect x="9" y="9" width="11" height="11" rx="2"></rect>'
        '<path d="M14 9V7a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h2"></path>'
        "</svg>"
        '<svg class="ok" viewBox="0 0 24 24" aria-hidden="true">'
        '<path d="M5 13l4 4L19 7"></path>'
        "</svg>"
        "</button>"
        if has_letter
        else ""
    )
    ai_chip = '<span class="ai" title="AI/LLM-вакансия">AI</span>' if vacancy.is_ai_priority else ""
    return f"""
      <li class="row">
        <div class="num" aria-hidden="true">{position:02d}</div>
        <div class="cell">
          <div class="head">
            <a class="title" href="{html.escape(vacancy.url, quote=True)}" target="_blank" rel="noopener">
              {html.escape(vacancy.title)}
            </a>
            {copy_button}
          </div>
          <p class="meta">
            <span class="company">{company}</span>
            <span class="date">{_short_date(vacancy.discovered_at)}</span>
            {ai_chip}
          </p>
          {attention_block}
          {letter}
          <div class="actions">
            <a class="act done" data-action href="{base}?as=applied">Откликнулся</a>
            <a class="act skip" data-action href="{base}?as=skipped">Пропустить</a>
          </div>
        </div>
      </li>"""


STYLE = """<style>
  :root {
    --paper: #ffffff;
    --ink: #15191d;
    --muted: #6c7781;
    --line: #e4e7ea;
    --link: #16538c;
    --ok: #17694a;
    --ok-ink: #ffffff;
    --skip: #8f2b2b;
    --warn-ink: #8a4b00;
    --warn-rule: #d8962f;
    --hover: #f7f8f9;
    color-scheme: light dark;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --paper: #111416;
      --ink: #e8ecef;
      --muted: #8e9aa5;
      --line: #262b30;
      --link: #7fb4e8;
      --ok: #2f8f63;
      --ok-ink: #f4fff9;
      --skip: #e59a9a;
      --warn-ink: #e7c089;
      --warn-rule: #a8762a;
      --hover: #171b1e;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 0 20px 64px;
    background: var(--paper);
    color: var(--ink);
    font: 16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
      "Helvetica Neue", Arial, sans-serif;
    -webkit-text-size-adjust: 100%;
  }
  .wrap { max-width: 660px; margin: 0 auto; }

  header {
    position: sticky;
    top: 0;
    z-index: 2;
    background: var(--paper);
    padding: 26px 0 12px;
    border-bottom: 1px solid var(--line);
  }
  .headline { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; }
  h1 {
    margin: 0;
    font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    font-size: clamp(28px, 7vw, 36px);
    font-weight: 600;
    letter-spacing: -0.015em;
    line-height: 1.1;
  }
  .count {
    font-variant-numeric: tabular-nums;
    font-size: 15px;
    color: var(--muted);
    white-space: nowrap;
  }
  .hint { margin: 6px 0 0; font-size: 13px; color: var(--muted); }

  ul { list-style: none; margin: 0; padding: 0; }
  .row {
    display: grid;
    grid-template-columns: 34px 1fr;
    gap: 10px;
    padding: 18px 4px;
    border-bottom: 1px solid var(--line);
    transition: opacity .16s ease, transform .16s ease;
  }
  @media (hover: hover) {
    .row:hover { background: var(--hover); }
  }
  .row.leaving { opacity: 0; transform: translateX(10px); }
  .num {
    font-variant-numeric: tabular-nums;
    font-size: 13px;
    color: var(--muted);
    padding-top: 3px;
    letter-spacing: 0.03em;
  }
  .title {
    display: inline-block;
    font-size: 17px;
    font-weight: 600;
    line-height: 1.35;
    color: var(--link);
    text-decoration: none;
    padding: 2px 0;
  }
  .title:hover { text-decoration: underline; text-underline-offset: 2px; }
  .meta { margin: 4px 0 0; font-size: 13.5px; color: var(--muted); display: flex; flex-wrap: wrap; gap: 0 10px; }
  .ai {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: var(--paper);
    background: var(--ink);
    border-radius: 4px;
    padding: 1px 6px 2px;
    align-self: center;
  }
  .attention {
    margin: 9px 0 0;
    padding: 1px 0 1px 12px;
    border-left: 2px solid var(--warn-rule);
    font-size: 14px;
    line-height: 1.5;
    color: var(--ink);
  }
  .attention span {
    font-weight: 700;
    color: var(--warn-ink);
    margin-right: 7px;
  }
  .attention span::after { content: ":"; }
  .actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
  .act {
    font: inherit;
    font-size: 14px;
    line-height: 1;
    padding: 11px 16px;
    border-radius: 8px;
    border: 1px solid var(--line);
    background: transparent;
    color: var(--ink);
    text-decoration: none;
    cursor: pointer;
    min-height: 40px;
    display: inline-flex;
    align-items: center;
  }
  .act:focus-visible { outline: 2px solid var(--link); outline-offset: 2px; }
  .done { background: var(--ok); border-color: var(--ok); color: var(--ok-ink); font-weight: 600; }
  .skip { color: var(--skip); }
  .head { display: flex; align-items: center; gap: 10px; }
  .head .title { flex: 1 1 auto; }
  .copy {
    flex: 0 0 auto;
    width: 34px;
    height: 34px;
    margin-top: 1px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: transparent;
    color: var(--muted);
    cursor: pointer;
  }
  .copy:hover { color: var(--ink); border-color: var(--muted); }
  .copy:focus-visible { outline: 2px solid var(--link); outline-offset: 2px; }
  .copy svg { width: 16px; height: 16px; fill: none; stroke: currentColor; stroke-width: 1.8;
              stroke-linecap: round; stroke-linejoin: round; }
  .copy .ok { display: none; color: var(--ok); }
  .copy.copied .ico { display: none; }
  .copy.copied .ok { display: block; }

  .empty { padding: 56px 0; text-align: center; }
  .empty strong { display: block; font-size: 18px; margin-bottom: 6px; }
  .empty p { margin: 0; color: var(--muted); font-size: 14px; }

  footer { margin-top: 34px; display: flex; justify-content: flex-end; }
  .clear {
    font: inherit;
    font-size: 13px;
    color: var(--muted);
    background: none;
    border: 0;
    padding: 8px 4px;
    cursor: pointer;
    text-decoration: underline;
    text-underline-offset: 3px;
  }
  .clear:hover { color: var(--skip); }
  @media (prefers-reduced-motion: reduce) {
    .row, .row.leaving { transition: none; }
  }
  @media (max-width: 420px) {
    body { padding-inline: 16px; }
    .row { grid-template-columns: 26px 1fr; gap: 8px; }
    .actions .act { flex: 1 1 auto; justify-content: center; }
  }
</style>"""

SCRIPT = """<script>
(function () {
  function pad(value) { return value < 10 ? '0' + value : String(value); }

  function refreshPositions() {
    var rows = document.querySelectorAll('li.row');
    rows.forEach(function (row, index) {
      var num = row.querySelector('.num');
      if (num) num.textContent = pad(index + 1);
    });
    var count = document.getElementById('count');
    if (count) count.textContent = String(rows.length);
    var empty = document.getElementById('empty');
    if (empty) empty.hidden = rows.length > 0;
  }

  function copyText(text) {
    if (window.isSecureContext && navigator.clipboard) {
      navigator.clipboard.writeText(text);
      return true;
    }
    var area = document.createElement('textarea');
    area.value = text;
    area.setAttribute('readonly', '');
    area.style.position = 'fixed';
    area.style.top = '-1000px';
    document.body.appendChild(area);
    area.select();
    area.setSelectionRange(0, text.length);
    var copied = false;
    try { copied = document.execCommand('copy'); } catch (error) { copied = false; }
    area.remove();
    return copied;
  }

  document.addEventListener('click', function (event) {
    var copyButton = event.target.closest('button.copy');
    if (copyButton) {
      var source = document.getElementById(copyButton.dataset.target);
      if (!source) return;
      var copied = copyText(source.textContent);
      copyButton.classList.toggle('copied', copied);
      copyButton.title = copied ? 'Скопировано' : 'Не удалось скопировать';
      setTimeout(function () {
        copyButton.classList.remove('copied');
        copyButton.title = 'Скопировать письмо';
      }, 1500);
      return;
    }

    var action = event.target.closest('a[data-action]');
    if (!action) return;
    event.preventDefault();
    var row = action.closest('li.row');
    fetch(action.href, { headers: { 'X-Requested-With': 'fetch' } })
      .then(function (response) {
        if (!response.ok) { window.location = action.href; return; }
        if (!row) { window.location = action.href; return; }
        row.classList.add('leaving');
        setTimeout(function () {
          row.remove();
          refreshPositions();
        }, 170);
      })
      .catch(function () { window.location = action.href; });
  });

  var clearForm = document.querySelector('form.clear');
  if (clearForm) {
    clearForm.addEventListener('submit', function (event) {
      if (!window.confirm('Снять с очереди все вакансии? Они получат статус «пропущено».')) {
        event.preventDefault();
      }
    });
  }
})();
</script>"""


def render_page(vacancies: Sequence[Vacancy], *, token: str, message: str = "") -> str:
    rows = "\n".join(
        _row(vacancy, token, index) for index, vacancy in enumerate(vacancies, start=1)
    )
    note = ""
    if message == "applied":
        note = "Отмечено: отклик отправлен."
    elif message == "skipped":
        note = "Отмечено: пропущено."
    elif message == "cleared":
        note = "Очередь очищена."
    note_block = f'<p class="hint" role="status">{note}</p>' if note else ""
    clear_form = (
        f"""
      <footer>
        <form class="clear" method="post" action="/q/{html.escape(token)}/clear">
          <button class="clear" type="submit">Очистить очередь</button>
        </form>
      </footer>"""
        if vacancies
        else ""
    )
    empty_block = (
        '<div class="empty" id="empty" hidden>'
        "<strong>Очередь пуста.</strong>"
        "<p>Новые вакансии появятся здесь после следующего поиска.</p>"
        "</div>"
    )
    hint = (
        "Ссылка действует час. Отметки сразу меняют статус вакансии в боте."
    )
    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>Очередь: {len(vacancies)}</title>
{STYLE}
</head>
<body>
<div class="wrap">
  <header>
    <div class="headline">
      <h1>Очередь</h1>
      <span class="count" id="count">{len(vacancies)}</span>
    </div>
    <p class="hint">{hint}</p>
    {note_block}
  </header>

  {empty_block}

  <ul>
{rows}
  </ul>
{clear_form}
</div>
{SCRIPT}
</body>
</html>"""


class QueuePageServer:
    def __init__(self, settings: Settings, database: Database):
        self.settings = settings
        self.database = database
        self._runner: web.AppRunner | None = None

    @property
    def enabled(self) -> bool:
        return self.settings.queue_page_enabled and bool(self.settings.queue_page_public_url)

    def url(self) -> str:
        if not self.enabled:
            return ""
        base = self.settings.queue_page_public_url.rstrip("/")
        return f"{base}/q/{issue_token(self.settings.queue_page_secret)}"

    def _token_valid(self, token: str) -> bool:
        return token_is_valid(self.settings.queue_page_secret, token)

    async def handle_page(self, request: web.Request) -> web.Response:
        token = request.match_info["token"]
        if not self._token_valid(token):
            raise web.HTTPForbidden(text="Ссылка устарела. Нажмите «Очередь» в боте ещё раз.")
        return web.Response(
            text=render_page(
                self.database.pending(), token=token, message=request.query.get("msg", "")
            ),
            content_type="text/html",
        )

    async def handle_action(self, request: web.Request) -> web.Response:
        token = request.match_info["token"]
        if not self._token_valid(token):
            raise web.HTTPForbidden(text="Ссылка устарела. Нажмите «Очередь» в боте ещё раз.")
        job_id = request.match_info["job_id"]
        action = request.query.get("as", "skipped")
        user_id = self.settings.tg_user_id
        if action == "applied":
            recorded = self.database.mark_manual_applied(job_id, user_id)
        else:
            recorded = self.database.skip(job_id, user_id, user_id)
        logger.info(
            "queue_page_action job_id=%s action=%s recorded=%s", job_id, action, recorded
        )
        if request.headers.get("X-Requested-With") == "fetch":
            return web.Response(status=204)
        message = "applied" if action == "applied" else "skipped"
        raise web.HTTPSeeOther(location=f"/q/{token}?msg={message}")

    async def handle_clear(self, request: web.Request) -> web.Response:
        token = request.match_info["token"]
        if not self._token_valid(token):
            raise web.HTTPForbidden(text="Ссылка устарела. Нажмите «Очередь» в боте ещё раз.")
        user_id = self.settings.tg_user_id
        cleared = 0
        for vacancy in self.database.pending():
            if self.database.skip(vacancy.id, user_id, user_id):
                cleared += 1
        logger.info("queue_page_cleared count=%s", cleared)
        raise web.HTTPSeeOther(location=f"/q/{token}?msg=cleared")

    def build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/q/{token}", self.handle_page)
        app.router.add_get("/q/{token}/done/{job_id}", self.handle_action)
        app.router.add_post("/q/{token}/clear", self.handle_clear)
        return app

    async def start(self) -> bool:
        if not self.settings.queue_page_enabled:
            return False
        runner = web.AppRunner(self.build_app())
        await runner.setup()
        site = web.TCPSite(
            runner, self.settings.queue_page_host, self.settings.queue_page_port
        )
        await site.start()
        self._runner = runner
        logger.info(
            "queue_page_started host=%s port=%s",
            self.settings.queue_page_host,
            self.settings.queue_page_port,
        )
        return True

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
