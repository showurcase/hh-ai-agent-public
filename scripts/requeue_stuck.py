#!/usr/bin/env python3
"""Вернуть в очередь откликов вакансии, застрявшие на инфраструктурных сбоях.

Речь о вакансиях, которые не дошли до отклика из-за таймаута LLM на анкете,
нераспознанной капчи или сетевого обрыва. Такие сбои не связаны с самой
вакансией, поэтому её безопасно попробовать ещё раз.

Без аргументов скрипт только показывает, что будет сделано.
С `--apply` он выполняет изменение.

    python scripts/requeue_stuck.py            # посмотреть
    python scripts/requeue_stuck.py --apply    # вернуть в очередь
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path("/app/agent.db")

# Причины, которые означают технический сбой, а не отказ вакансии.
TRANSIENT_REASONS = (
    "llm_error:timeout",
    "llm_error:transient_server",
    "captcha_not_solved",
    "Request timeout error",
)

# Статусы, из которых вакансия вообще может вернуться в работу.
STUCK_STATUSES = ("apply_failed", "pending_approval", "applying")


def find_stuck(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    placeholders = ", ".join("?" for _ in STUCK_STATUSES)
    conditions = " OR ".join("error_text LIKE ?" for _ in TRANSIENT_REASONS)
    sql = f"""
        SELECT id, title, company, status, error_text
        FROM vacancies
        WHERE status IN ({placeholders}) AND ({conditions})
        ORDER BY discovered_at
    """
    patterns = [f"%{reason}%" for reason in TRANSIENT_REASONS]
    return connection.execute(sql, (*STUCK_STATUSES, *patterns)).fetchall()


def requeue(connection: sqlite3.Connection, job_ids: list[str]) -> int:
    if not job_ids:
        return 0
    placeholders = ", ".join("?" for _ in job_ids)
    cursor = connection.execute(
        f"""
        UPDATE vacancies SET
            status = 'retry_pending',
            apply_retry_count = 0,
            approved_at = NULL,
            applying_at = NULL,
            approver_id = NULL,
            permit_hash = NULL
        WHERE id IN ({placeholders})
          AND status IN ({', '.join('?' for _ in STUCK_STATUSES)})
        """,
        (*job_ids, *STUCK_STATUSES),
    )
    return cursor.rowcount


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="выполнить изменение; без флага скрипт только показывает список",
    )
    parser.add_argument("--db", default=str(DB_PATH), help="путь к базе")
    args = parser.parse_args()

    connection = sqlite3.connect(args.db)
    connection.row_factory = sqlite3.Row
    rows = find_stuck(connection)
    if not rows:
        print("Застрявших вакансий нет.")
        return 0

    print(f"Найдено вакансий: {len(rows)}")
    for row in rows:
        print(f"  {row['id']}  {row['status']:<16} {row['title']} — {row['company']}")
        print(f"      причина: {row['error_text'] or 'нет'}")

    if not args.apply:
        print("\nЭто только просмотр. Для возврата в очередь запустите с --apply.")
        return 0

    changed = requeue(connection, [row["id"] for row in rows])
    connection.commit()
    print(f"\nВернули в очередь откликов: {changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
