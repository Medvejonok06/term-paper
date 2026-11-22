"""Єдина схема ознак: використовується і при навчанні, і в API (без train/serve розбіжностей)."""
import numpy as np
import pandas as pd

from src.features.dixon_coles import expected_goals, predict_score_matrix
from src.features.elo import HOME_ADVANTAGE

PRIOR_PPG = 1.35

# Ковзні статистики команди (EWMA по ВСІХ її матчах): {показник: вікна}
ROLL_SPECS: dict[str, tuple[int, ...]] = {
    "xgf": (3, 5, 10), "xga": (3, 5, 10), "pts": (5, 10),
    "gf": (5,), "ga": (5,), "sotf": (5,), "sota": (5,),
    "cs": (10,), "fts": (10,), "fin": (10,),
}
PRIORS = {"xgf": 1.35, "xga": 1.35, "pts": 1.35, "gf": 1.35, "ga": 1.35,
          "sotf": 4.5, "sota": 4.5, "cs": 0.25, "fts": 0.25, "fin": 0.0}

# Форма саме у ролі господаря / гостя
VENUE_STATS = ("pts", "gf", "ga")
VENUE_SPAN = 6
VENUE_PRIORS = {True: {"pts": 1.6, "gf": 1.5, "ga": 1.15}, False: {"pts": 1.15, "gf": 1.15, "ga": 1.5}}

SCHED_KEYS = ("rest", "g14", "g30")
NEUTRAL_SCHEDULE = {"rest": 7.0, "g14": 2.0, "g30": 4.0}   # для гіпотетичного матчу
TABLE_KEYS = ("ppg", "rank", "gdpg")


def team_keys() -> list[str]:
    return [f"{stat}_{span}" for stat, spans in ROLL_SPECS.items() for span in spans]


def venue_keys() -> list[str]:
    return [f"v_{s}" for s in VENUE_STATS]


def _both(keys) -> list[str]:
    return [f"{side}_{k}" for k in keys for side in ("home", "away")]


def _uniq(cols: list[str]) -> list[str]:
    return list(dict.fromkeys(cols))


DC_COLS = ["dc_prob_home", "dc_prob_draw", "dc_prob_away"]
SHIN_COLS = ["shin_prob_home", "shin_prob_draw", "shin_prob_away"]

CORE = ["elo_home", "elo_away", "elo_diff", "xg_diff_5", "pts_diff_5",
        "exp_home_goals", "exp_away_goals"] + DC_COLS
ORIG_ROLL = ["xgf_3", "xgf_5", "xgf_10", "xga_3", "xga_5", "xga_10", "pts_5", "pts_10"]
EXT_ROLL = ["gf_5", "ga_5", "sotf_5", "sota_5", "cs_10", "fts_10", "fin_10"]
DIFFS = ["sot_diff_5", "fin_diff_10", "venue_pts_diff", "venue_gd_diff", "rest_diff", "rank_diff", "ppg_diff"]
H2H_COLS = ["h2h_n", "h2h_ppg", "h2h_gd"]

# Набори ознак для порівняння (добір за валідацією)
V1_COLS = _uniq(CORE + _both(ORIG_ROLL))
LEAN_COLS = _uniq(CORE + DIFFS + ["h2h_ppg", "h2h_gd", "home_g14", "away_g14", "season_progress"])
FULL_COLS = _uniq(V1_COLS + _both(EXT_ROLL) + _both(venue_keys()) + _both(SCHED_KEYS)
                  + _both(TABLE_KEYS) + ["season_progress"] + DIFFS + H2H_COLS)

FUND_SETS = {"v1": V1_COLS, "lean": LEAN_COLS, "full": FULL_COLS}
MARKET_SETS = {"market_lean": _uniq(SHIN_COLS + LEAN_COLS), "market_full": _uniq(SHIN_COLS + FULL_COLS)}

BASE_COLS = V1_COLS                                   # сумісність зі старим кодом
MARKET_COLS = _uniq(FULL_COLS + SHIN_COLS)            # усе, що зберігаємо в parquet


def h2h_features(results: list) -> tuple[float, float, float]:
    """results: [(голи_господарів, голи_гостей), ...] з погляду господаря, останні ≤5 зустрічей."""
    n = len(results)
    pts = sum(3 if gf > ga else 1 if gf == ga else 0 for gf, ga in results)
    gd = sum(gf - ga for gf, ga in results)
    k = 2.0                                           # зглажування до середнього
    return float(n), (pts + PRIOR_PPG * k) / (n + k), gd / (n + k)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Похідні ознаки. Та сама функція для датасету і для одного кастомного матчу."""
    df = df.copy()
    df["elo_diff"] = df["elo_home"] - df["elo_away"] + HOME_ADVANTAGE
    df["xg_diff_5"] = (df["home_xgf_5"] - df["home_xga_5"]) - (df["away_xgf_5"] - df["away_xga_5"])
    df["pts_diff_5"] = df["home_pts_5"] - df["away_pts_5"]
    df["sot_diff_5"] = (df["home_sotf_5"] - df["home_sota_5"]) - (df["away_sotf_5"] - df["away_sota_5"])
    df["fin_diff_10"] = df["home_fin_10"] - df["away_fin_10"]
    df["venue_pts_diff"] = df["home_v_pts"] - df["away_v_pts"]
    df["venue_gd_diff"] = (df["home_v_gf"] - df["home_v_ga"]) - (df["away_v_gf"] - df["away_v_ga"])
    df["rest_diff"] = df["home_rest"] - df["away_rest"]
    df["rank_diff"] = df["away_rank"] - df["home_rank"]       # > 0: господарі вище в таблиці
    df["ppg_diff"] = df["home_ppg"] - df["away_ppg"]

    lam, mu = expected_goals(df["home_xgf_10"], df["home_xga_10"], df["away_xgf_10"], df["away_xga_10"])
    df["exp_home_goals"], df["exp_away_goals"] = lam, mu
    probs = np.array([predict_score_matrix(l, m)[:3] for l, m in zip(lam, mu)])
    for i, col in enumerate(DC_COLS):
        df[col] = probs[:, i]
    return df


def matchup_from_state(home: dict, away: dict) -> pd.DataFrame:
    """Рядок ознак для гіпотетичного матчу зі збережених станів команд.
    Календар (відпочинок) нейтральний, таблиця береться на момент останнього зіграного матчу."""
    row = {"elo_home": home["elo"], "elo_away": away["elo"], "season_progress": 0.5}
    for key in team_keys():
        row[f"home_{key}"], row[f"away_{key}"] = home[key], away[key]
    for key in venue_keys():
        row[f"home_{key}"], row[f"away_{key}"] = home["venue_home"][key], away["venue_away"][key]
    for key in SCHED_KEYS:
        row[f"home_{key}"] = row[f"away_{key}"] = NEUTRAL_SCHEDULE[key]
    for key in TABLE_KEYS:
        row[f"home_{key}"], row[f"away_{key}"] = home["table"][key], away["table"][key]
    n, ppg, gd = h2h_features([tuple(x) for x in home.get("h2h", {}).get(away["name"], [])])
    row.update({"h2h_n": n, "h2h_ppg": ppg, "h2h_gd": gd})
    return add_derived(pd.DataFrame([row]))