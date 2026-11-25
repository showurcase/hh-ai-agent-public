"""Показать, что происходит с капчей hh.ru: кнопка языка и картинка до/после клика.

Запуск при остановленном боте:
    docker compose stop
    docker compose run --rm --entrypoint python \
        -v ~/hh-ai-agent/scripts:/app/scripts:ro \
        hh-agent /app/scripts/probe_captcha.py <url>
    docker compose start
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from browser_backend import create_browser_backend  # noqa: E402
from config import load_settings  # noqa: E402

LANGUAGE_SELECTORS = (
    'button[data-qa="captcha-language"]',
    'button:has-text("English")',
    'button:has-text("Русский")',
    'a:has-text("English")',
)
PICTURE_SELECTORS = (
    'img[data-qa="account-captcha-picture"]',
    'img[alt="captcha"]',
    'img[src*="/captcha/picture"]',
)


async def describe(page, label: str) -> None:
    print(f"--- {label} ---", flush=True)
    print("url:", str(page.url)[:120], flush=True)
    for selector in LANGUAGE_SELECTORS:
        locator = page.locator(selector).first
        try:
            visible = await locator.is_visible()
        except Exception as exc:
            print(f"  {selector}: error {type(exc).__name__}", flush=True)
            continue
        if not visible:
            print(f"  {selector}: hidden", flush=True)
            continue
        text = aria = ""
        try:
            text = (await locator.inner_text()).strip()
        except Exception:
            pass
        try:
            aria = await locator.get_attribute("aria-label") or ""
        except Exception:
            pass
        print(f"  {selector}: text={text!r} aria={aria!r}", flush=True)
    for selector in PICTURE_SELECTORS:
        locator = page.locator(selector).first
        try:
            visible = await locator.is_visible()
        except Exception:
            continue
        if not visible:
            continue
        src = await locator.get_attribute("src")
        print(f"  {selector}: visible src={str(src)[:70]!r}", flush=True)


async def main(urls: list[str]) -> None:
    settings = load_settings()
    backend = create_browser_backend(settings)
    context = await backend.start()
    page = None
    try:
        page = await context.new_page()
        for url in urls:
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            await asyncio.sleep(3)
            control = page.locator('button[data-qa="captcha-language"]').first
            if not await control.is_visible():
                print("no captcha at", url, flush=True)
                continue
            await describe(page, f"captcha at {url}")
            await page.screenshot(path="/tmp/captcha-before.png")
            await control.click()
            await asyncio.sleep(3)
            await describe(page, "after first click")
            await page.screenshot(path="/tmp/captcha-after1.png")
            await control.click()
            await asyncio.sleep(3)
            await describe(page, "after second click")
            await page.screenshot(path="/tmp/captcha-after2.png")
            return
        print("no captcha found", flush=True)
    finally:
        if page is not None:
            await page.close()
        await backend.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: probe_captcha.py <url> [url ...]")
    asyncio.run(main(sys.argv[1:]))
