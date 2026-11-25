"""Забрать cookies hh.ru из уже открытого Chrome через отладочный порт.

Chrome должен быть запущен с --remote-debugging-port=9222 и тем же профилем,
в котором владелец вошёл в аккаунт:

    python export_cookies_cdp.py /tmp/hh-cookies.json
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from playwright.async_api import async_playwright


async def main(out: Path) -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.connect_over_cdp(
            "http://127.0.0.1:9222"
        )
        cookies = []
        logged_in = False
        for context in browser.contexts:
            for cookie in await context.cookies():
                if "hh.ru" not in cookie.get("domain", ""):
                    continue
                cookies.append(cookie)
                if cookie.get("name") == "hhrole" and cookie.get("value") not in (
                    "",
                    "anonymous",
                ):
                    logged_in = True
        print(f"cookies hh.ru: {len(cookies)}, признак входа: {logged_in}", flush=True)
        if not cookies:
            raise SystemExit("cookies hh.ru не найдены — войдите в открытом окне")
        out.write_text(json.dumps(cookies, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"сохранено в {out}", flush=True)
        await browser.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: export_cookies_cdp.py <out.json>")
    asyncio.run(main(Path(sys.argv[1])))
