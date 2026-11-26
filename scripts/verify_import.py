"""Перевірка відтворюваності: тимчасова БД -> імпорт CSV -> звірка з робочою БД -> видалення.
Облікові дані беруться з налаштувань (.env), пароль нікуди не виводиться.
  python -m scripts.verify_import"""
from __future__ import annotations

import os
import subprocess
import sys

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from src.config import settings

TEST_DB = "football_import_test"
CHECKS = {
    "teams": "SELECT COUNT(*) FROM teams",
    "matches": "SELECT COUNT(*), SUM(home_score), SUM(away_score) FROM matches",
    "match_stats": "SELECT COUNT(*), ROUND(CAST(SUM(home_xg) AS numeric), 3) FROM match_stats",
}


def snapshot(url) -> dict:
    eng = create_engine(url)
    try:
        with eng.connect() as c:
            return {k: tuple(c.execute(text(q)).one()) for k, q in CHECKS.items()}
    finally:
        eng.dispose()


def main() -> None:
    url = make_url(settings.DATABASE_URL)
    test_url = url.set(database=TEST_DB)
    admin = create_engine(url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as c:
            c.execute(text(f"DROP DATABASE IF EXISTS {TEST_DB}"))
            c.execute(text(f"CREATE DATABASE {TEST_DB}"))
    except Exception as e:
        raise SystemExit(f"❌ Не вдалося створити тимчасову БД (потрібне право CREATEDB): {type(e).__name__}")
    try:
        env = {**os.environ, "DATABASE_URL": test_url.render_as_string(hide_password=False)}
        subprocess.run([sys.executable, "-m", "scripts.import_db"], check=True, env=env)
        a, b = snapshot(url), snapshot(test_url)
        for k in CHECKS:
            print(f"{k:<12} робоча={a[k]}  імпорт={b[k]}  {'✅' if a[k] == b[k] else '❌'}")
        print("✅ Імпорт відтворює дані" if a == b else "❌ Дані розійшлись")
    finally:
        with admin.connect() as c:
            c.execute(text(f"DROP DATABASE IF EXISTS {TEST_DB}"))
        admin.dispose()


if __name__ == "__main__":
    main()
