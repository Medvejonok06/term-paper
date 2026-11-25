"""Створює таблиці й завантажує дані з data/db_export/*.csv у порожню БД.
  python -m scripts.import_db       (адреса БД береться з DATABASE_URL або .env)"""
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.db.base import Base
from src.db.models import Match, MatchStats, Team
from src.db.session import engine

SRC = Path("data/db_export")
ORDER = (Team, Match, MatchStats)


def main() -> None:
    Base.metadata.create_all(engine)
    with engine.connect() as conn:
        for model in ORDER:
            t = model.__tablename__
            if conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar():
                raise SystemExit(f"Таблиця {t} не порожня, імпорт скасовано")
    for model in ORDER:
        t = model.__tablename__
        df = pd.read_csv(SRC / f"{t}.csv").convert_dtypes()
        df.to_sql(t, engine, if_exists="append", index=False)
        print(f"{t}: завантажено {len(df)}")
    for model in ORDER:                                   # вирівнюємо лічильники id після явного завантаження
        t = model.__tablename__
        try:
            with engine.begin() as conn:
                conn.execute(text(f"SELECT setval(pg_get_serial_sequence('{t}', 'id'), COALESCE((SELECT MAX(id) FROM {t}), 1))"))
        except Exception:
            pass


if __name__ == "__main__":
    main()
