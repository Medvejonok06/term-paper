"""Синхронізація БД із football-data.co.uk: нові матчі, нові команди, xG-проксі з ударів.
  python -m src.data.sync_football_data                    # поточний сезон
  python -m src.data.sync_football_data --from-year 2014   # плюс старіші сезони (бекфіл)"""
from __future__ import annotations

import argparse
import io
from datetime import date

import numpy as np
import pandas as pd
import requests
from sqlalchemy import Date as SADate
from sqlalchemy import DateTime as SADateTime
from sqlalchemy import inspect as sa_inspect

from src.db.models import Match, MatchStats, Team
from src.db.session import SessionLocal

URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
STATS_MAP = {"home_shots": "HS", "away_shots": "AS", "home_shots_target": "HST", "away_shots_target": "AST",
             "b365_home_win_odds": "B365H", "b365_draw_odds": "B365D", "b365_away_win_odds": "B365A"}
KNOWN_MATCH = {"season", "match_date", "home_team_id", "away_team_id", "home_score", "away_score", "result_1x2"}
KNOWN_STATS = set(STATS_MAP) | {"match_id", "home_xg", "away_xg"}


def current_start_year() -> int:
    t = date.today()
    return t.year if t.month >= 7 else t.year - 1


def unknown_required(model, known: set[str]) -> list[str]:
    return [c.name for c in sa_inspect(model).columns
            if not c.nullable and not c.primary_key and c.default is None and c.server_default is None
            and c.name not in known]


def download(year: int) -> pd.DataFrame | None:
    code = f"{year % 100:02d}{(year + 1) % 100:02d}"
    r = requests.get(URL.format(code=code), timeout=30, headers={"User-Agent": "epl-predictor-coursework/1.0"})
    if r.status_code != 200:
        print(f"  {year}/{code}: HTTP {r.status_code}, пропускаю")
        return None
    df = pd.read_csv(io.BytesIO(r.content), encoding_errors="replace")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    df["match_date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    return df


def fit_xg_proxy(db) -> tuple[float, float]:
    """xG ≈ a·(удари в площину) + b·(інші удари), підігнано на матчах, де xG є в БД."""
    X, y = [], []
    for s in db.query(MatchStats).all():
        for xg, sh, sot in ((s.home_xg, s.home_shots, s.home_shots_target),
                            (s.away_xg, s.away_shots, s.away_shots_target)):
            if xg and xg > 0 and sh is not None and sot is not None:
                X.append([sot, max(sh - sot, 0)])
                y.append(xg)
    if len(y) < 200:
        return 0.30, 0.04
    coef, *_ = np.linalg.lstsq(np.array(X, float), np.array(y, float), rcond=None)
    return float(max(coef[0], 0.05)), float(max(coef[1], 0.0))


def coerce_date(ts: pd.Timestamp):
    t = sa_inspect(Match).columns["match_date"].type
    if isinstance(t, SADateTime):
        return ts.to_pydatetime()
    if isinstance(t, SADate):
        return ts.date()
    return ts.strftime("%Y-%m-%d")


def sync(from_year: int, to_year: int) -> None:
    for model, known in ((Match, KNOWN_MATCH), (MatchStats, KNOWN_STATS)):
        bad = unknown_required(model, known)
        if bad:
            raise SystemExit(f"{model.__name__} має обов'язкові поля, які я не заповнюю: {bad}. Скинь models.py.")

    with SessionLocal() as db:
        teams = {t.name: t.id for t in db.query(Team).all()}
        keys = {(str(d)[:10], h, a) for d, h, a in db.query(Match.match_date, Match.home_team_id, Match.away_team_id)}
        a, b = fit_xg_proxy(db)
        print(f"xG-проксі: {a:.3f}·SOT + {b:.3f}·(удари−SOT)")

        total = 0
        for year in range(from_year, to_year + 1):
            df = download(year)
            if df is None:
                continue
                label, added = f"{year % 100:02d}{(year + 1) % 100:02d}", 0
            for r in df.to_dict("records"):
                for name in (r["HomeTeam"], r["AwayTeam"]):
                    if name not in teams:
                        t = Team(name=name)
                        db.add(t)
                        db.flush()
                        teams[name] = t.id
                        print(f"  нова команда: {name}")
                h, aw = teams[r["HomeTeam"]], teams[r["AwayTeam"]]
                key = (r["match_date"].strftime("%Y-%m-%d"), h, aw)
                if key in keys:
                    continue
                hg, ag = int(r["FTHG"]), int(r["FTAG"])
                m = Match(season=label, match_date=coerce_date(r["match_date"]), home_team_id=h, away_team_id=aw,
                          home_score=hg, away_score=ag, result_1x2="H" if hg > ag else "D" if hg == ag else "A")
                db.add(m)
                db.flush()

                vals = {}
                for field, col in STATS_MAP.items():
                    v = r.get(col)
                    if hasattr(MatchStats, field) and v is not None and not pd.isna(v):
                        vals[field] = int(v) if "shots" in field else float(v)
                for side, hs, hst in (("home", "HS", "HST"), ("away", "AS", "AST")):
                    sh, sot = r.get(hs), r.get(hst)
                    if hasattr(MatchStats, f"{side}_xg") and pd.notna(sh) and pd.notna(sot):
                        vals[f"{side}_xg"] = max(0.05, a * float(sot) + b * max(float(sh) - float(sot), 0.0))
                db.add(MatchStats(match_id=m.id, **vals))
                keys.add(key)
                added += 1
            print(f"{label}: додано {added} матчів")
            total += added
        db.commit()
    print(f"✅ Разом додано: {total}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--from-year", type=int, default=current_start_year())
    p.add_argument("--to-year", type=int, default=current_start_year())
    args = p.parse_args()
    sync(args.from_year, args.to_year)