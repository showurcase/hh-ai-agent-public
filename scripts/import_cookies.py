"""Загрузить cookies hh.ru из экспортированного JSON в профиль бота.

Файл берётся у владельца аккаунта: расширение Cookie-Editor на hh.ru → Export → JSON.
Запуск при остановленном боте:
    docker compose stop
    docker compose run --rm --entrypoint python \
        -v ~/hh-ai-agent/scripts:/app/scripts:ro \
        -v /tmp/hh-cookies.json:/tmp/hh-cookies.json:ro \
        hh-agent /app/scripts/import_cookies.py /tmp/hh-cookies.json
    docker compose start
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from approval import ApprovalGuard  # noqa: E402
from browser_backend import create_browser_backend  # noqa: E402
from config import load_settings  # noqa: E402
from database import Database  # noqa: E402
from hh_client import HHClient  # noqa: E402


def to_playwright_cookies(raw: list[dict]) -> list[dict]:
    cookies = []
    for item in raw:
        domain = item.get("domain") or ""
        if "hh.ru" not in domain:
            continue
        cookie = {
            "name": item["name"],
            "value": item["value"],
            "domain": domain,
            "path": item.get("path") or "/",
        }
        if item.get("expirationDate"):
            cookie["expires"] = float(item["expirationDate"])
        if item.get("secure") is not None:
            cookie["secure"] = bool(item["secure"])
        if item.get("httpOnly") is not None:
            cookie["httpOnly"] = bool(item["httpOnly"])
        same_site = item.get("sameSite")
        if same_site in {"Strict", "Lax", "None"}:
            cookie["sameSite"] = same_site
        cookies.append(cookie)
    return cookies


async def main(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("cookies", [])
    cookies = to_playwright_cookies(payload)
    print(f"cookies для hh.ru: {len(cookies)}", flush=True)
    if not cookies:
        raise SystemExit("в файле нет cookies для hh.ru")

    settings = load_settings()
    database = Database(settings.database_path)
    database.init()
    backend = create_browser_backend(settings)
    context = await backend.start()
    try:
        await context.add_cookies(cookies)
        print("cookies добавлены в профиль", flush=True)
        client = HHClient(
            context, settings, database, ApprovalGuard(settings, database)
        )
        print("ensure_login:", await client.ensure_login(), flush=True)
    finally:
        await backend.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: import_cookies.py <cookies.json>")
    asyncio.run(main(Path(sys.argv[1])))
