import re


_HORIZONTAL_RULE = re.compile(r"^[ \t]*(?:-{3,}|_{3,}|\*{3,})[ \t]*$")
_PUNCTUATION_ONLY = re.compile(r"^[ \t]*[.…。·•,:;!?-]+[ \t]*$")
_MARKDOWN_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)")
_COMPLETE_ENDING = re.compile(r'(?:[.!?…][»”"\')\]]*|[🙂👋])$')
# A short final line such as "+" or a code word is a reply marker, not a cut sentence.
_MARKER_LINE_MAX = 20


def letter_urls(text: str) -> set[str]:
    return {
        match.rstrip(".,;:!?")
        for match in re.findall(r"(?<![\w/@])(?:https?://|www\.)[^\s<>\[\]()\"']+", text)
    }


def has_required_portfolio(text: str, required_url: str) -> bool:
    return not required_url or required_url in letter_urls(text)


def _plain_markdown_link(match: re.Match[str]) -> str:
    label = match.group(1).strip()
    url = match.group(2)
    return url if label.rstrip("/") == url.rstrip("/") else label



def humanize_text(text: str) -> str:
    """Убирает маркеры нейросетевого текста (AI-slop): тире, скобки, слэши, штампы."""
    if not text:
        return ""

    # 0. Защищаем URL
    urls = []
    def save_url(m):
        urls.append(m.group(0))
        return f"__URL_{len(urls)-1}__"
    cleaned = re.sub(r"https?://\S+", save_url, text)

    # 1. ПОЛНЫЙ ЗАПРЕТ на длинные (—) и средние (–) тире
    # Тире с пробелами заменяем на запятую
    cleaned = re.sub(r"\s+[—–]\s+", ", ", cleaned)
    # Одиночные тире внутри слов/конструкций заменяем на обычный дефис
    cleaned = cleaned.replace("—", "-").replace("–", "-")

    # 2. Единицы измерения со слэшем: пользователей/сутки -> пользователей в сутки
    per_units = r"(?:сутки|суток|сек|сек\.|секунду|мин|мин\.|минуту|час|часов|мес|мес\.|месяц|день|дня|дней|год|года|лет)"
    cleaned = re.sub(rf"([а-яА-Яa-zA-Z0-9]+)\s*/\s*({per_units})\b", r"\1 в \2", cleaned, flags=re.I)

    # 3. Слэши между терминами/словами: OpenAPI/Swagger -> OpenAPI и Swagger
    cleaned = re.sub(r"([a-zA-Zа-яА-Я0-9]+)\s*/\s*([a-zA-Zа-яА-Я0-9]+)", r"\1 и \2", cleaned)

    # 4. Скобки: убираем или распаковываем в естественную речь
    # (например, X) -> , например X,
    cleaned = re.sub(r"\(\s*(например[^)]+)\)", r", \1,", cleaned, flags=re.I)
    # (net) / (gross) -> на руки / до вычета
    cleaned = re.sub(r"\(\s*net\s*\)", "на руки", cleaned, flags=re.I)
    cleaned = re.sub(r"\(\s*gross\s*\)", "до вычета налогов", cleaned, flags=re.I)
    # Обычные скобки-уточнения: раскрываем запятыми
    cleaned = re.sub(r"\s*\(([^)]+)\)", r", \1,", cleaned)

    # 5. Убираем нейро-штампы и вводную воду
    cliches = [
        (r"\b(стоит|важно|следует) отметить,?\s*(что)?\s*", ""),
        (r"\bкроме того,?\s*", ""),
        (r"\bв частности,?\s*", ""),
        (r"\bкак синхронные,?\s*так и асинхронные\b", "синхронные и асинхронные"),
        (r"\bне только\b([^\n.]+?)\bно и\b", r"\1 и"),
        (r"\bактивно\s+", ""),
        (r"\bрегулярно\s+", ""),
        (r"\bдетально\s+", ""),
        (r"\bбесшовн(ый|ая|ое|ые|ую|ых|ым|ыми)\b", r"надежн\1"),
        (r"\bглубок(ий|ая|ое|ую|ие|их|им|ими)\s+экспертиз\w*", "опыт"),
        (r"\bв рамках проекта\b", "в проекте"),
        (r"\bв рамках работы\b", "в работе"),
        (r"\bв рамках задачи\b", "в задаче"),
        (r"\bв рамках задач\b", "в задачах"),
        (r"\bв рамках команды\b", "в команде"),
        (r"\bна сегодняшний день\b", "сейчас"),
        (r"\bв настоящее время\b", "сейчас"),
        (r"\bв связи с этим,?\s*", ""),
        (r"\bтаким образом,?\s*", ""),
        (r"\bв заключение,?\s*", ""),
        (r"\bширокий спектр\b", "разные"),
        (r"\bнеотъемлемая часть\b", "часть"),
        (r"\bиграет важную роль\b", "важен"),
        (r"\bуспешно\s+", ""),
        (r"\bэффективн\w+\s+", ""),
    ]
    for pattern, repl in cliches:
        cleaned = re.sub(pattern, repl, cleaned, flags=re.I)

    # 6. Чистим дублирующую пунктуацию
    cleaned = re.sub(r",\s*,+", ",", cleaned)
    cleaned = re.sub(r",\s*\.", ".", cleaned)
    cleaned = re.sub(r"\.\s*,", ".", cleaned)
    cleaned = re.sub(r"\s+([.,!?:;])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # 7. Восстанавливаем заглавные буквы в начале предложений, если удалили вводные слова
    lines = []
    for line in cleaned.splitlines():
        line = re.sub(r"(^|[.!?]\s+)([а-яa-z])", lambda m: m.group(1) + m.group(2).upper(), line)
        lines.append(line)
    cleaned = "\n".join(lines)

    # 8. Возвращаем URL
    for idx, u in enumerate(urls):
        cleaned = cleaned.replace(f"__URL_{idx}__", u)

    return cleaned.strip()


def normalize_cover_letter_body(
    letter: str, required_url: str = "", closing: str = ""
) -> str:
    """Convert an LLM draft into clean plain text without changing its prose."""
    body = letter.replace("\r\n", "\n").replace("\r", "\n").strip()
    body = _MARKDOWN_LINK.sub(_plain_markdown_link, body)
    if closing and body.rstrip().endswith(closing):
        body = body.rstrip()[: -len(closing)].rstrip()
    if required_url:
        body = re.sub(
            r"(?im)^[ \t]*Портфолио:[ \t]*" + re.escape(required_url) + r"[ \t]*$",
            "",
            body,
        )
        body = body.replace(required_url, "")
    body = re.sub(r"(?is)(?:\n\s*Контакты(?: для связи)?:\s*)?(?:Telegram|Телеграм|TG):[ \t]*\S+\s*$", "", body).rstrip()
    body = re.sub(r"(?is)\n\s*С уважением[,\s].*$", "", body).rstrip()

    clean_lines: list[str] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if _HORIZONTAL_RULE.fullmatch(line) or _PUNCTUATION_ONLY.fullmatch(line):
            continue
        if not line:
            if clean_lines and clean_lines[-1]:
                clean_lines.append("")
            continue
        clean_lines.append(line)
    while clean_lines and not clean_lines[-1]:
        clean_lines.pop()
    return humanize_text("\n".join(clean_lines).strip())


def _body_without_footer(text: str, required_url: str, closing: str) -> str:
    body = text.strip()
    if closing and body.endswith(closing):
        body = body[: -len(closing)].rstrip()
    if required_url:
        body = re.sub(
            r"(?im)^[ \t]*Портфолио:[ \t]*" + re.escape(required_url) + r"[ \t]*$",
            "",
            body,
        ).strip()
    body = re.sub(r"(?is)(?:\n\s*Контакты(?: для связи)?:\s*)?(?:Telegram|Телеграм|TG):[ \t]*\S+\s*$", "", body).rstrip()
    body = re.sub(r"(?is)\n\s*С уважением[,\s].*$", "", body).rstrip()
    return body


def _ends_completely(body: str) -> bool:
    stripped = body.rstrip()
    if _COMPLETE_ENDING.search(stripped):
        return True
    last_line = stripped.splitlines()[-1].strip() if stripped else ""
    return (
        bool(last_line)
        and len(last_line) <= _MARKER_LINE_MAX
        and " " not in last_line
        and "," not in last_line
    )


def cover_letter_validation_error(
    text: str, required_url: str, closing: str, max_length: int
) -> str:
    if not text.strip():
        return "cover_letter_empty"
    if len(text) > max_length:
        return "cover_letter_too_long"
    if "```" in text or _MARKDOWN_LINK.search(text):
        return "cover_letter_markdown"
    for line in text.splitlines():
        if _HORIZONTAL_RULE.fullmatch(line) or _PUNCTUATION_ONLY.fullmatch(line):
            return "cover_letter_layout_artifact"
    if not has_required_portfolio(text, required_url):
        return "required_portfolio_missing"
    if closing and not text.rstrip().endswith(closing):
        return "required_closing_missing"
    body = _body_without_footer(text, required_url, closing)
    if not body:
        return "cover_letter_body_empty"
    if (required_url or closing) and not _ends_completely(body):
        return "cover_letter_incomplete"
    return ""


def is_sendable_cover_letter(
    text: str, required_url: str, closing: str, max_length: int
) -> bool:
    return not cover_letter_validation_error(
        text, required_url, closing, max_length
    )


def format_portfolio_footer(
    letter: str, required_url: str, closing: str, max_length: int
) -> str:
    """Normalize a draft and append the exact footer without truncating prose."""
    body = normalize_cover_letter_body(letter, required_url, closing)
    if not body:
        return ""
    footer_parts = []
    if required_url:
        footer_parts.append(f"Портфолио: {required_url}")
    if closing:
        footer_parts.append(closing)
    result = "\n\n".join([body, *footer_parts])
    if cover_letter_validation_error(result, required_url, closing, max_length):
        return ""
    return result
