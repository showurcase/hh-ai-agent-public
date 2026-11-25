from cover_letter import humanize_text


def test_humanize_removes_machine_phrases_and_layout() -> None:
    raw = (
        "В рамках проекта это неотъемлемая часть. "
        "Таким образом, мы успешно сделали API — и базу (PostgreSQL)."
    )

    cleaned = humanize_text(raw)

    assert "В рамках" not in cleaned
    assert "неотъемлемая" not in cleaned
    assert "Таким образом" not in cleaned
    assert "успешно" not in cleaned
    assert "—" not in cleaned
    assert "(" not in cleaned
    assert "API, и базу" in cleaned


def test_humanize_keeps_urls_and_terms() -> None:
    raw = (
        "Смотрите https://example.com/profile — там OpenAPI/Swagger "
        "и PostgreSQL."
    )

    cleaned = humanize_text(raw)

    assert "https://example.com/profile" in cleaned
    assert "OpenAPI и Swagger" in cleaned
    assert "PostgreSQL" in cleaned


def test_telegram_link_closing_is_appended_exactly() -> None:
    from cover_letter import format_portfolio_footer

    closing = (
        "С уважением,\nИван Петров\nTelegram: https://t.me/example"
    )

    letter = format_portfolio_footer("Короткое письмо.", "", closing, 1000)

    assert letter.endswith(closing)
    assert "https://t.me/example" in letter
