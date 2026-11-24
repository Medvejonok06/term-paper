"""Журнал живих прогнозів: фіксуємо прогноз ДО початку матчу (час запису зберігається).
  python -m src.live.log_predictions              # найближчий тур
  python -m src.live.log_predictions --rounds 2   # і наступний
Потрібен запущений бекенд (uvicorn на порту 8000)."""
from __future__ import annotations

import argparse
import io
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.features.shin_odds import shin_de_margin

API = "http://127.0.0.1:8000/api/v1"
LOG = Path("data/live/predictions.jsonl")
ODDS_URL = "https://www.football-data.co.uk/fixtures.csv"   # якщо адреса змінилась: логуватимемо без коефіцієнтів


def fetch_odds() -> dict[tuple[str, str], list[float]]:
    try:
        r = requests.get(ODDS_URL, timeout=20, headers={"User-Agent": "epl-predictor-coursework/1.0"})
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content), encoding_errors="replace")
    except Exception as e:
        print(f"Коефіцієнти недоступні ({e}), логую без них")
        return {}
    if "Div" in df.columns:
        df = df[df["Div"] == "E0"]
    out = {}
    for r in df.to_dict("records"):
        try:
            p = shin_de_margin(*[float(r[c]) for c in ("B365H", "B365D", "B365A")])
        except (KeyError, TypeError, ValueError):
            continue
        if all(pd.notna(v) for v in p):
            out[(r["HomeTeam"], r["AwayTeam"])] = [float(v) for v in p]
    return out


def model_stamp() -> str:
    parts = [f"{p.name}:{int(p.stat().st_mtime)}" for p in sorted(Path("models").glob("bundle_*.joblib"))]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "n/a"
    return f"{commit}|{';'.join(parts)}"


def main(rounds: int) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if LOG.exists():
        for line in LOG.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                seen.add((d["home"], d["away"], d["season"]))

    odds, now, stamp = fetch_odds(), datetime.now(timezone.utc), model_stamp()
    event, new = None, 0
    for _ in range(rounds):
        url = f"{API}/fixtures/next" if event is None else f"{API}/fixtures/round/{event}"
        r = requests.get(url, timeout=120)
        if not r.ok:
            raise SystemExit(f"Не вдалося отримати тур: {r.status_code} {r.text[:200]}")
        rnd = r.json()
        event = rnd["event"] + 1
        for f in rnd["fixtures"]:
            if not f.get("kickoff") or f.get("prob_home_win") is None:
                continue
            kick = datetime.fromisoformat(f["kickoff"].replace("Z", "+00:00"))
            if kick <= now:                                  # матч уже почався: прогноз був би нечесним
                continue
            season = kick.year if kick.month >= 7 else kick.year - 1
            home, away = f["home_team"], f["away_team"]
            if (home, away, season) in seen:
                continue
            pr = requests.post(f"{API}/predict/custom", json={"home_team": home, "away_team": away,
                                                              "adjust_absences": True}, timeout=120)
            if not pr.ok:
                print(f"  пропускаю {home} - {away}: {pr.status_code}")
                continue
            p = pr.json()
            adj = p.get("adjustment") or {"home_elo": 0.0, "away_elo": 0.0}
            probs = [p["prob_home_win"], p["prob_draw"], p["prob_away_win"]]
            rec = {"logged_at": now.isoformat(), "season": season, "event": rnd["event"], "kickoff": f["kickoff"],
                   "home": home, "away": away, "probs_adj": probs, "probs_base": p.get("base_probs") or probs,
                   "elo_home_base": p["home_context"]["elo"] - adj["home_elo"],
                   "elo_away_base": p["away_context"]["elo"] - adj["away_elo"],
                   "adjustment": adj, "bookmaker": odds.get((home, away)), "model_stamp": stamp}
            with LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            seen.add((home, away, season))
            new += 1
            print(f"  {home} - {away}: " + " / ".join(f"{v:.0%}" for v in probs))
    print(f"✅ Нових прогнозів: {new}. Усього в журналі: {len(seen)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    main(ap.parse_args().rounds)