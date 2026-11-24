"""Живий бектест: порівнюємо прогнози, зафіксовані ДО матчів, з реальними результатами (FPL).
  python -m src.live.evaluate"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src.data import fpl
from src.features.elo import HOME_ADVANTAGE
from src.ml.ensemble import rps_score

LOG = Path("data/live/predictions.jsonl")
OUT = Path("data/live/summary.json")
FEATURES = Path("data/processed/epl_features.parquet")
STATE = Path("data/processed/team_state.json")


def load_settled() -> tuple[int, list[dict]]:
    rows = [json.loads(l) for l in LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    season = max(r["season"] for r in rows)
    rows = [r for r in rows if r["season"] == season]
    ids = fpl.name_map(json.loads(STATE.read_text(encoding="utf-8")).keys())
    results = {}
    for f in fpl.fixtures() or []:
        if (f.get("finished") or f.get("finished_provisional")) and f.get("team_h_score") is not None:
            h, a = ids.get(f["team_h"]), ids.get(f["team_a"])
            if h and a:
                results[(h, a)] = (f["team_h_score"], f["team_a_score"])
    settled = []
    for r in rows:
        s = results.get((r["home"], r["away"]))
        if s:
            r["y"] = 0 if s[0] > s[1] else 1 if s[0] == s[1] else 2
            settled.append(r)
    return len(rows), settled


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    return {"accuracy": round(float((p.argmax(1) == y).mean() * 100), 2),
            "log_loss": round(float(log_loss(y, p, labels=[0, 1, 2])), 4), "rps": round(rps_score(y, p), 4)}


def main() -> None:
    if not LOG.exists():
        raise SystemExit("Журнал порожній: спершу python -m src.live.log_predictions")
    logged, settled = load_settled()
    n = len(settled)
    if n == 0:
        raise SystemExit(f"Зафіксовано прогнозів: {logged}, завершених матчів серед них ще немає.")

    df = pd.read_parquet(FEATURES).dropna(subset=["target"])
    fit = df[df["split"].isin(["train", "val"])]               # ті самі дані, що й у заморожених моделей
    yfit = fit["target"].astype(int)
    elo_lr = LogisticRegression().fit(fit[["elo_diff"]], yfit)
    freq = np.bincount(yfit, minlength=3) / len(yfit)

    y = np.array([r["y"] for r in settled])
    elo_diff = pd.DataFrame({"elo_diff": [r["elo_home_base"] - r["elo_away_base"] + HOME_ADVANTAGE for r in settled]})
    adj, base = "Модель (з урахуванням травм)", "Модель (без урахування травм)"
    P = {adj: np.array([r["probs_adj"] for r in settled]), base: np.array([r["probs_base"] for r in settled]),
         "Elo-регресія": elo_lr.predict_proba(elo_diff), "Частоти результатів": np.tile(freq, (n, 1))}
    has = np.array([r.get("bookmaker") is not None for r in settled])
    book = np.array([r["bookmaker"] if r.get("bookmaker") else [1 / 3] * 3 for r in settled])

    models = [{"name": k, "n": n, **scores(y, p)} for k, p in P.items()]
    if has.any():
        models.append({"name": "Букмекери (Shin, лише матчі з коефіцієнтами)", "n": int(has.sum()),
                       **scores(y[has], book[has])})

    rng = np.random.default_rng(0)

    def per(p, yy):
        return -np.log(np.clip(p[np.arange(len(yy)), yy], 1e-9, 1))

    def compare(name, a, b, yy):
        d = per(a, yy) - per(b, yy)
        item = {"name": name, "n": int(len(d)), "diff": round(float(d.mean()), 4), "lo": None, "hi": None}
        if len(d) >= 20:
            boots = [rng.choice(d, len(d)).mean() for _ in range(5000)]
            item["lo"], item["hi"] = [round(float(v), 4) for v in np.percentile(boots, [2.5, 97.5])]
        return item

    cmps = [compare("з урахуванням травм − без", P[adj], P[base], y),
            compare("модель − Elo-регресія", P[adj], P["Elo-регресія"], y)]
    if has.any():
        cmps.append(compare("модель − букмекери", P[adj][has], book[has], y[has]))

    summary = {"logged": logged, "settled": n, "with_odds": int(has.sum()), "models": models, "comparisons": cmps}
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Зафіксовано: {logged}, завершено: {n}, з коефіцієнтами: {int(has.sum())}")
    print(f"{'Модель':<48}{'n':>5}{'Acc%':>8}{'LogLoss':>9}{'RPS':>8}")
    for m in models:
        print(f"{m['name']:<48}{m['n']:>5}{m['accuracy']:>8.2f}{m['log_loss']:>9.4f}{m['rps']:>8.4f}")
    print("\nРізниця log loss (мінус = модель краща):")
    for c in cmps:
        ci = f"95% ДІ [{c['lo']:+.4f}, {c['hi']:+.4f}]" if c["lo"] is not None else "n<20, ДІ не рахуємо"
        print(f"  {c['name']:<30} n={c['n']:<4} {c['diff']:+.4f}  {ci}")


if __name__ == "__main__":
    main()