"""Открыть браузер, дать владельцу войти на hh.ru и сохранить cookies в JSON.

Запускается на машине владельца (не в контейнере):
    python login_and_export.py /tmp/hh-cookies.json

Проверка входа не перезагружает страницу: сначала смотрим на текущую вкладку,
затем на cookie hhrole, и только потом один раз идём на /applicant/resumes.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from playwright.async_api import async_playwright

LOGIN_MARKERS = (
    '[data-qa="mainmenu_applicantProfile"], [data-qa="mainmenu_profile"], '
    '[data-qa="mainmenu_resumes"], [data-qa="resume-title"]'
)


async def looks_logged_in(page, context) -> bool:
    try:
        if await page.locator(LOGIN_MARKERS).count() > 0:
            return True
    except Exception:
        pass
    try:
        for cookie in await context.cookies():
            if cookie.get("name") == "hhrole" and cookie.get("value") not in (
                "",
                "anonymous",
            ):
                return True
    except Exception:
        pass
    return False


async def confirm(page) -> bool:
    try:
        await page.goto(
            "https://hh.ru/applicant/resumes",
            wait_until="domcontentloaded",
            timeout=30_000,
        )
    except Exception:
        return False
    if "account/login" in str(page.url):
        return False
    try:
        return await page.locator(LOGIN_MARKERS).count() > 0
    except Exception:
        return False


async def main(out: Path) -> None:
    async with async_playwright() as playwright:
        context = await playwright.chromium.launch_persistent_context(
            user_data_dir="/tmp/hh-login-profile",
            channel="chrome",
            headless=False,
        )
        page = context.pages[0] if context.pages else await context.new_page()
        await page.goto("https://hh.ru/", wait_until="domcontentloaded")
        print("Войдите в аккаунт hh.ru в открывшемся окне.", flush=True)

        for _ in range(300):
            await asyncio.sleep(2)
            if not await looks_logged_in(page, context):
                continue
            if await confirm(page):
                cookies = [
                    cookie
                    for cookie in await context.cookies()
                    if "hh.ru" in cookie.get("domain", "")
                ]
                out.write_text(
                    json.dumps(cookies, ensure_ascii=False, indent=1), encoding="utf-8"
                )
                print(f"Вход подтверждён. Cookies: {len(cookies)} -> {out}", flush=True)
                await context.close()
                return
        print("Вход не выполнен за 10 минут.", flush=True)
        await context.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: login_and_export.py <out.json>")
    asyncio.run(main(Path(sys.argv[1])))
