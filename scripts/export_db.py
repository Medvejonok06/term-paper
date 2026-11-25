"""Експорт БД у CSV (data/db_export/), щоб проєкт відтворювався без доступу до вашого сервера Postgres.
  python -m scripts.export_db"""
from pathlib import Path

import pandas as pd

from src.db.models import Match, MatchStats, Team
from src.db.session import engine

OUT = Path("data/db_export")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for model in (Team, Match, MatchStats):
        name = model.__tablename__
        df = pd.read_sql_table(name, engine)
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} рядків")


if __name__ == "__main__":
    main()
