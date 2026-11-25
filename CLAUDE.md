# HH AI Agent — коротко для агентов

Бот ищет вакансии на hh.ru, оценивает их LLM-моделью, пишет сопроводительное письмо и откликается.
Управление — Telegram-бот. Очередь на ручное решение — локальная веб-страница.

Отвечай на языке пользователя. Персональные данные кандидата живут только в `profile.yaml` и `legend/` — в git их нет.

## LLM

Весь LLM-трафик идёт через **OpenCode Go** (`https://opencode.ai/inference/go/openai/v1`), это OpenAI-совместимый endpoint. Основная модель — `deepseek-v4.1-flash`, она же используется для vision-капчи (`CAPTCHA_MODEL`). Провайдер требует заголовок `x-opencode-session`, бот ставит его сам. Ключ только в `.env`, в коде и git его нет.

## Где что запускается

| | Где | Что это |
|---|---|---|
| Прод | сервер с Docker, каталог проекта | Рабочая копия, из неё поднимается контейнер `hh-ai-agent` через `docker compose` |
| Репозиторий | GitHub, remote `origin` | История и резервная копия |

На сервере git-истории может не быть: там клон и незакоммиченные файлы. **Никогда** не делай на сервере
`git checkout .`, `git reset --hard`, `git clean` — это сотрёт все доработки без возможности восстановить.

Сервер задаётся своим SSH-алиасом (например `myserver`) в `~/.ssh/config`.

## Как деплоить

Почти все рабочие модули примонтированы в контейнер отдельными файлами (`docker-compose.yml`).
Docker следует за inode, поэтому обычный `rsync` (он пересоздаёт файл) оставит контейнер со старым кодом.

```bash
rsync -a --inplace main.py hh_client.py myserver:/path/to/hh-ai-agent/
```

Дальше:

- изменился `.py` из списка монтирований → `docker compose restart`
- изменился `.env` или `docker-compose.yml` → `docker compose up -d --force-recreate`
- **новый файл** `.py` → сначала добавить его в `volumes:` в `docker-compose.yml`, иначе он не доедет
  (так уже было с `logging_setup.py`: правка «задеплоилась», а контейнер работал со старым кодом из образа)

## Тесты

```bash
rsync -a --inplace --exclude '.git' --exclude '.env' --exclude 'agent.db*' \
  --exclude 'cookies.json' --exclude '.browser-profile' --exclude 'legend' \
  --exclude 'backups' --exclude '.pytest_cache' ./ myserver:/path/to/hh-ai-agent-test/

cd /path/to/hh-ai-agent-test && bash scripts/run_tests.sh
```

`scripts/run_tests.sh` поднимает контейнер, доставляет `git` и браузер Playwright (без них часть
тестов падает по причинам окружения) и прогоняет весь набор. Перед деплоем в прод набор должен быть
полностью зелёным.

## Карта кода

| Файл | Зачем |
|---|---|
| `main.py` | Цикл поиска, обработка вакансии, расписание пачек, `process_vacancy` |
| `hh_client.py` | Всё взаимодействие с hh.ru: поиск, чтение вакансии, отклик, анкеты, капча |
| `ai_analyzer.py` | Промпты и вызовы LLM: пригодность, письмо, ответы на анкету, капча |
| `approval.py` | Право на отклик: `auto_apply`, удержание AI-вакансий, ручной отклик |
| `database.py` | SQLite: вакансии, статусы, счётчики, флаги бота |
| `tg_bot.py` | Telegram: меню, карточки вакансий, команды |
| `queue_page.py` | Веб-страница очереди на 0.0.0.0:8787 |
| `ai_priority.py` | Признаки AI/LLM в тексте вакансии |
| `cover_letter.py` | Валидация и «очеловечивание» письма |
| `config.py` | Настройки из `.env` и `profile.yaml` |
| `legend.py`, `questionnaire.py` | Легенда кандидата и заполнение анкет работодателя |
| `browser_backend.py` | Запуск CloakBrowser/Playwright |
| `logging_setup.py` | Логи; файл примонтирован, ротация копированием |
| `scripts/probe_*.py` | Ручные пробы: `probe_read.py`, `probe_captcha.py`, `probe_letter.py`, `probe_apply.py` |

## Как работает отклик

1. `run_search_cycle` обходит запросы из `profile.yaml` (`hh.search_queries`); глубина и объём —
   `MAX_PAGES_PER_QUERY` и `MAX_VACANCIES_PER_QUERY`.
2. По каждой: читает страницу → жёсткие фильтры профиля → оценка LLM (`assess`) → письмо → карточка.
3. Если включён флаг `manual_review_disabled` (по умолчанию **да**, то есть «всё автоматом»), отклик уходит
   сам через `ApprovalService.auto_apply`: разрешение → проверка дневного лимита → клик «Откликнуться» →
   анкета → капча → финальная кнопка → подтверждение на странице.

Статусы (`VacancyStatus`): `discovered` → `rejected_by_filter` / `rejected_by_llm` / `pending_approval` →
`approved` → `applying` → `applied` / `apply_failed`. Отдельно: `retry_pending` (внутренняя очередь повтора),
`skipped` (ответ уже есть, вакансия снята, чужая страна без удалёнки, недоступна аккаунту), `expired`.

Повтор после сбоя: первая неудача отклика → `retry_pending`, бот пробует ещё раз в начале следующего цикла;
вторая → `pending_approval` и карточка владельцу. Вакансии, упавшие до письма, перечитываются заново.

AI/LLM-вакансии помечаются `is_ai_priority`. Если `manual_review_disabled` выключить, такие вакансии
перестают откликаться автоматом и приходят карточкой владельцу на ручной отклик.

## Что владелец делает руками

Telegram-меню (`/menu`): «Поиск сейчас» (принудительный цикл), «Пауза/Запустить», «Режим»,
«Статистика» (там же счётчик AI-откликов), «Очередь» (ссылка на страницу), «Ручная работа» (переключатель),
«Поднять резюме». Карточка вакансии: «Откликнуться», «Пропустить», копирование письма.

## Правила, которые нельзя нарушать

- **Секреты не коммитить:** `.env`, `cookies.json`, `agent.db*`, `profile.yaml`, `legend/`, `.browser-profile*`,
  `*.log` — они в `.gitignore`. Не добавляй их через `git add -f`.
- **Не пушить в чужой upstream.** Работа уходит только в свой `origin`.
- На сервере прод-база `agent.db`: перед правкой схемы делай `cp agent.db agent.db.backup-$(date +%Y%m%d%H%M)`.
- Правка схемы статусов в SQLite требует пересборки таблицы — это делает `_migrate_status_check` при старте.

## Если бот разлогинился на hh.ru

Признаки: в логе `hh_login_required`, все отклики падают с «final application button was not found»,
на странице вакансии вместо формы отклика — блок «Напишите телефон». В шапке сайта при этом «Войти».

Как вернуть сессию (на сервере графики нет, поэтому вход делается на машине владельца):

1. На Маке открыть чистый Chrome: `open -na "Google Chrome" --args --user-data-dir=/tmp/hh-login-profile --remote-debugging-port=9222 https://hh.ru/`
2. Владелец входит в аккаунт в этом окне.
3. Забрать cookies: `python scripts/export_cookies_cdp.py /tmp/hh-cookies.json` (venv с playwright).
4. Перенести файл на сервер и залить в профиль бота при остановленном контейнере:
   `docker compose run --rm --entrypoint python -v ./scripts:/app/scripts:ro -v /tmp/hh-cookies.json:/tmp/hh-cookies.json:ro hh-agent /app/scripts/import_cookies.py /tmp/hh-cookies.json`
5. Скрипт печатает `ensure_login: True` — вход восстановлен. Временные файлы cookies удалить.

Не открывай страницу с проверкой входа в цикле с перезагрузкой: это делает логин невозможным.

## Где смотреть, когда сломалось

```bash
docker logs hh-ai-agent --since 1h 2>&1 | grep -aE "^2026-" | grep -aiE "ERROR|failed|paused"
docker exec hh-ai-agent python -c "import sqlite3;d=sqlite3.connect('/app/agent.db');\
print(list(d.execute('select status, count(*) from vacancies group by status')))"
```

Частые причины: капча (HH принимает ответ только с русской картинки: `CAPTCHA_SWITCH_TO_ENGLISH=false`;
если всё же переключать — кнопка языка это тумблер, её подпись называет язык, на который переключит),
дневной лимит откликов, протухшая сессия hh.ru (страница логина), `Вам недоступна эта вакансия`.

Дневной лимит и окно работы — `MAX_APPLICATIONS_PER_DAY`, `AUTO_APPLY_START_HOUR` / `AUTO_APPLY_END_HOUR`
(`AUTO_APPLY_TIMEZONE`, по факту `Europe/Moscow`).
