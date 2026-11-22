"""БД -> Elo, Shin, форма, домашня/виїзна форма, календар, таблиця, H2H, Dixon-Coles -> parquet + стан команд."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.db.models import Match, MatchStats, Team
from src.db.session import SessionLocal
from src.features.elo import MovEloCalculator
from src.features.schema import (
    MARKET_COLS, PRIOR_PPG, PRIORS, ROLL_SPECS, SCHED_KEYS, TABLE_KEYS, VENUE_PRIORS, VENUE_SPAN,
    VENUE_STATS, add_derived, h2h_features, team_keys, venue_keys,
)
from src.features.shin_odds import shin_de_margin

OUT_DIR = Path("data/processed")


def _num(x) -> float:
    return float(x) if x is not None else np.nan


def _opt(obj, name: str) -> float:
    """Необов'язкове поле: якщо колонки в моделі БД немає, повертаємо NaN."""
    return _num(getattr(obj, name, None))


def load_raw() -> tuple[pd.DataFrame, dict[int, str]]:
    with SessionLocal() as db:
        names = {t.id: t.name for t in db.query(Team).all()}
        rows = (db.query(Match, MatchStats).join(MatchStats, Match.id == MatchStats.match_id)
                .order_by(Match.match_date, Match.id).all())
        records = [{
            "match_id": m.id, "match_date": m.match_date, "season": m.season,
            "home_team_id": m.home_team_id, "away_team_id": m.away_team_id,
            "home_score": m.home_score, "away_score": m.away_score, "result_1x2": m.result_1x2,
            "home_xg": _num(s.home_xg), "away_xg": _num(s.away_xg),
            "home_sot": _opt(s, "home_shots_target"), "away_sot": _opt(s, "away_shots_target"),
            "b365h": _num(s.b365_home_win_odds), "b365d": _num(s.b365_draw_odds),
            "b365a": _num(s.b365_away_win_odds),
        } for m, s in rows]
    df = pd.DataFrame(records)
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df.sort_values(["match_date", "match_id"]).reset_index(drop=True), names


def season_context(df: pd.DataFrame):
    """Турнірна таблиця та H2H ДО кожного матчу (без витоку майбутнього)."""
    n = len(df)
    cols = {c: np.zeros(n) for c in ("home_ppg", "away_ppg", "home_rank", "away_rank", "home_gdpg",
                                     "away_gdpg", "season_progress", "h2h_n", "h2h_ppg", "h2h_gd")}
    season_teams = {s: set(g.home_team_id) | set(g.away_team_id) for s, g in df.groupby("season")}
    table: dict[int, list] = {}      # team -> [очки, різниця, забито, зіграно]
    h2h: dict[tuple, list] = {}
    season = None

    for i, r in enumerate(df.itertuples(index=False)):
        if r.season != season:
            season = r.season
            table = {t: [0, 0, 0, 0] for t in season_teams[season]}
        h, a = r.home_team_id, r.away_team_id
        keys = {t: (-v[0], -v[1], -v[2]) for t, v in table.items()}

        for side, t in (("home", h), ("away", a)):
            pts, gd, _, played = table[t]
            cols[f"{side}_ppg"][i] = (pts + PRIOR_PPG * 3) / (played + 3)
            cols[f"{side}_gdpg"][i] = gd / (played + 3)
            cols[f"{side}_rank"][i] = 1 + sum(k < keys[t] for k in keys.values())
        cols["season_progress"][i] = (table[h][3] + table[a][3]) / 76.0

        pair = (min(h, a), max(h, a))
        past = [(hg, ag) if hid == h else (ag, hg) for hid, hg, ag in h2h.get(pair, [])[-5:]]
        cols["h2h_n"][i], cols["h2h_ppg"][i], cols["h2h_gd"][i] = h2h_features(past)

        hp = 3 if r.home_score > r.away_score else 1 if r.home_score == r.away_score else 0
        ap = {3: 0, 1: 1, 0: 3}[hp]
        for t, p, gf, ga in ((h, hp, r.home_score, r.away_score), (a, ap, r.away_score, r.home_score)):
            table[t][0] += p; table[t][1] += gf - ga; table[t][2] += gf; table[t][3] += 1
        h2h.setdefault(pair, []).append((h, r.home_score, r.away_score))
    return cols, table, h2h


def add_schedule(long: pd.DataFrame) -> None:
    """Дні відпочинку та кількість матчів за 14 і 30 днів (тільки матчі ліги)."""
    for c in SCHED_KEYS:
        long[c] = 0.0
    for _, idx in long.groupby("team_id").groups.items():
        days = long.loc[idx, "match_date"].values.astype("datetime64[D]").astype("int64")
        pos = np.arange(len(days))
        long.loc[idx, "rest"] = np.concatenate([[14.0], np.minimum(np.diff(days), 14)])
        long.loc[idx, "g14"] = pos - np.searchsorted(days, days - 14, side="left")
        long.loc[idx, "g30"] = pos - np.searchsorted(days, days - 30, side="left")


def build_features_dataset() -> None:
    df, names = load_raw()
    print(f"--> Матчів: {len(df)}")

    for side, goals in (("home_xg", "home_score"), ("away_xg", "away_score")):
        df[side] = df[side].where(df[side] > 0).fillna(df[goals])      # xG<=0 = пропуск

    shin = np.array([shin_de_margin(h, d, a) for h, d, a in zip(df.b365h, df.b365d, df.b365a)])
    df["shin_prob_home"], df["shin_prob_draw"], df["shin_prob_away"] = shin.T

    elo = MovEloCalculator()
    elo_home, elo_away, prev_season = [], [], None
    first_season = df["season"].iloc[0]
    for r in df.itertuples():
        if prev_season is not None and r.season != prev_season:
            elo.new_season()
        prev_season = r.season
        for tid in (r.home_team_id, r.away_team_id):
            if tid not in elo.ratings and r.season != first_season:
                elo.ratings[tid] = 1400.0      # новачок після першого сезону слабший за середню команду
        elo_home.append(elo.get_rating(r.home_team_id))
        elo_away.append(elo.get_rating(r.away_team_id))
        elo.update_match(r.home_team_id, r.away_team_id, r.home_score, r.away_score)
    df["elo_home"], df["elo_away"] = elo_home, elo_away

    ctx, final_table, h2h = season_context(df)
    for name, values in ctx.items():
        df[name] = values

    # --- довга таблиця: рядок = команда у матчі ---
    def points(gf, ga):
        return np.where(gf > ga, 3, np.where(gf == ga, 1, 0))

    def side_frame(is_home: bool) -> pd.DataFrame:
        p, o = ("home", "away") if is_home else ("away", "home")
        gf, ga = df[f"{p}_score"], df[f"{o}_score"]
        xgf = df[f"{p}_xg"]
        return pd.DataFrame({
            "match_id": df.match_id, "match_date": df.match_date, "team_id": df[f"{p}_team_id"],
            "is_home": is_home, "gf": gf, "ga": ga, "xgf": xgf, "xga": df[f"{o}_xg"],
            "pts": points(gf, ga), "sotf": df[f"{p}_sot"], "sota": df[f"{o}_sot"],
            "cs": (ga == 0).astype(float), "fts": (gf == 0).astype(float), "fin": gf - xgf,
        })

    long = (pd.concat([side_frame(True), side_frame(False)])
            .sort_values(["match_date", "match_id"]).reset_index(drop=True))
    long["games_played"] = long.groupby("team_id").cumcount()

    for stat, spans in ROLL_SPECS.items():                      # форма по всіх матчах
        for span in spans:
            key = f"{stat}_{span}"
            full = long.groupby("team_id")[stat].transform(lambda s, sp=span: s.ewm(span=sp, adjust=False).mean())
            long[f"_full_{key}"] = full
            long[key] = full.groupby(long["team_id"]).shift(1).fillna(PRIORS[stat])

    for stat in VENUE_STATS:                                    # форма окремо вдома / на виїзді
        full = (long.groupby(["team_id", "is_home"])[stat]
                .transform(lambda s: s.ewm(span=VENUE_SPAN, adjust=False).mean()))
        long[f"_full_v_{stat}"] = full
        pre = full.groupby([long["team_id"], long["is_home"]]).shift(1)
        prior = pd.Series(np.where(long.is_home, VENUE_PRIORS[True][stat], VENUE_PRIORS[False][stat]),
                          index=long.index)
        long[f"v_{stat}"] = pre.fillna(prior)

    add_schedule(long)

    keys = ["games_played"] + team_keys() + venue_keys() + list(SCHED_KEYS)

    def side(is_home: bool, prefix: str) -> pd.DataFrame:
        part = long.loc[long.is_home == is_home, ["match_id", *keys]]
        return part.rename(columns={c: f"{prefix}_{c}" for c in keys})

    df = df.merge(side(True, "home"), on="match_id").merge(side(False, "away"), on="match_id")

    # --- поточний стан команд для кастомних прогнозів ---
    last = long.groupby("team_id").tail(1).set_index("team_id")
    last_home = long[long.is_home].groupby("team_id").tail(1).set_index("team_id")
    last_away = long[~long.is_home].groupby("team_id").tail(1).set_index("team_id")
    latest = df["season"].iloc[-1]
    active = set(df.loc[df.season == latest, "home_team_id"]) | set(df.loc[df.season == latest, "away_team_id"])

    tkeys = {t: (-v[0], -v[1], -v[2]) for t, v in final_table.items()}
    pair_hist: dict[str, dict[str, list]] = {}
    for (a, b), hist in h2h.items():
        for me, opp in ((a, b), (b, a)):
            res = [[hg, ag] if hid == me else [ag, hg] for hid, hg, ag in hist[-5:]]
            pair_hist.setdefault(names.get(me, str(me)), {})[names.get(opp, str(opp))] = res

    def venue_state(frame, team_id, is_home):
        return {f"v_{s}": float(frame.loc[team_id, f"_full_v_{s}"]) if team_id in frame.index
                else VENUE_PRIORS[is_home][s] for s in VENUE_STATS}

    state = {}
    for team_id, row in last.iterrows():
        name = names.get(team_id, f"Team {team_id}")
        if team_id in final_table:
            pts, gd, _, played = final_table[team_id]
            table_state = {"ppg": (pts + PRIOR_PPG * 3) / (played + 3), "gdpg": gd / (played + 3),
                           "rank": float(1 + sum(k < tkeys[team_id] for k in tkeys.values()))}
        else:
            table_state = {"ppg": 1.0, "gdpg": 0.0, "rank": 15.0}
        entry = {"team_id": int(team_id), "name": name, "elo": float(elo.get_rating(team_id)),
                 "active": team_id in active, "last_match": row.match_date.strftime("%Y-%m-%d"),
                 "venue_home": venue_state(last_home, team_id, True),
                 "venue_away": venue_state(last_away, team_id, False),
                 "table": table_state, "h2h": pair_hist.get(name, {})}
        entry.update({k: float(row[f"_full_{k}"]) for k in team_keys()})
        state[name] = entry

    df = add_derived(df)
    from src.features import dixon_coles as dc
    dc.set_calibration(xg_mean=(df["home_xgf_10"].mean() + df["away_xgf_10"].mean()) / 2,
                       home_goals=df["home_score"].mean(), away_goals=df["away_score"].mean())
    print(f"Калібрування голів: {dc.CALIB}")
    df["home_team"] = df["home_team_id"].map(names)
    df["away_team"] = df["away_team_id"].map(names)
    df["target"] = df["result_1x2"].map({"H": 0, "D": 1, "A": 2})

    meta = ["match_id", "match_date", "season", "home_team", "away_team", "home_score", "away_score",
            "result_1x2", "target", "home_games_played", "away_games_played"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df[meta + MARKET_COLS].to_parquet(OUT_DIR / "epl_features.parquet", index=False)
    (OUT_DIR / "team_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    nan_share = df[MARKET_COLS].isna().mean()
    print("Колонки з пропусками:", nan_share[nan_share > 0].round(3).to_dict() or "немає")
    print(f"✅ {len(df)} рядків, {len(MARKET_COLS)} ознак, стан {len(state)} команд -> {OUT_DIR}/")


if __name__ == "__main__":
    build_features_dataset()