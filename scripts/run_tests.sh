#!/usr/bin/env bash
# Прогон тестов в готовом образе. Образ собирается один раз и содержит git
# и браузер Playwright — без них часть тестов падает по причинам окружения.
#
# Запуск на сервере из каталога с кодом:
#   bash scripts/run_tests.sh
set -euo pipefail

image=hh-ai-agent-tests

if ! docker image inspect "$image" >/dev/null 2>&1; then
  echo "собираю образ для тестов (один раз)..."
  # В контейнерах нет IPv6, а DNS отдаёт IPv6-адреса первыми: сборка с сетью
  # хоста обходит «Network is unreachable» на pip и apt.
  docker build --network=host -f Dockerfile.test -t "$image" .
fi

docker run --rm -v "$PWD":/app -w /app "$image" python -m pytest tests -q
