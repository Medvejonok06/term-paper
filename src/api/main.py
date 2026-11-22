"""REST API v3: прогнози 1X2, матриця рахунків, склади/травми (FPL), календар туру, емблеми."""
from __future__ import annotations

import json
import logging
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from catboost import Pool
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from src.data import fpl
from src.features.dixon_coles import predict_score_matrix, top_scores
from src.features.schema import SHIN_COLS, matchup_from_state

log = logging.getLogger("uvicorn.error")
FEATURES_PATH = Path("data/processed/epl_features.parquet")
STATE_PATH = Path("data/processed/team_state.json")
MODELS_DIR = Path("models")
CODES = ["H", "D", "A"]
ART: dict = {}


# ---------- Схеми ----------
class TeamOption(BaseModel):
    name: str
    short: str
    crest: str | None = None


class ShapFactor(BaseModel):
    feature: str
    value: float | None
    impact: float


class ScoreProb(BaseModel):
    score: str
    prob: float


class TeamContext(BaseModel):
    name: str
    elo: float
    xgf_5: float
    xga_5: float
    pts_5: float
    crest: str | None = None


class Actual(BaseModel):
    score: str
    result: Literal["H", "D", "A"]
    correct: bool


class Adjustment(BaseModel):
    home_elo: float
    away_elo: float


class Prediction(BaseModel):
    match_id: int | None
    home_team: str
    away_team: str
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    predicted_class: Literal["H", "D", "A"]
    predicted_outcome: str
    model_used: Literal["market", "fundamental"]
    expected_home_goals: float
    expected_away_goals: float
    exact_scores: list[ScoreProb]
    score_matrix: list[list[float]]
    over_25: float
    btts: float
    shap_factors: list[ShapFactor]
    home_context: TeamContext
    away_context: TeamContext
    home_form: list[str]
    away_form: list[str]
    h2h: list[list[int]] | None = None
    bookmaker_probs: list[float] | None = None
    actual: Actual | None = None
    split: str | None = None
    adjustment: Adjustment | None = None
    data_until: str | None = None
    base_probs: list[float] | None = None


class CustomRequest(BaseModel):
    home_team: str
    away_team: str
    adjust_absences: bool = True


class MatchOut(BaseModel):
    id: int
    season: str
    match_date: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    result_1x2: str
    split: str


class FixtureOut(BaseModel):
    kickoff: str | None
    home_team: str
    away_team: str
    home_crest: str | None
    away_crest: str | None
    prob_home_win: float | None = None
    prob_draw: float | None = None
    prob_away_win: float | None = None
    predicted_class: Literal["H", "D", "A"] | None = None
    note: str | None = None


class NextRound(BaseModel):
    event: int
    fixtures: list[FixtureOut]


# ---------- Завантаження артефактів ----------
def _build_form(df: pd.DataFrame) -> dict[str, list]:
    to_home, to_away = {"H": "W", "D": "D", "A": "L"}, {"H": "L", "D": "D", "A": "W"}
    form: dict[str, list] = {}
    for r in df.sort_values(["match_date", "match_id"]).itertuples():
        if r.result_1x2 not in to_home:
            continue
        form.setdefault(r.home_team, []).append((r.match_date, to_home[r.result_1x2]))
        form.setdefault(r.away_team, []).append((r.match_date, to_away[r.result_1x2]))
    return form


@asynccontextmanager
async def lifespan(_: FastAPI):
    needed = [FEATURES_PATH, STATE_PATH, MODELS_DIR / "metrics.json",
              MODELS_DIR / "bundle_market.joblib", MODELS_DIR / "bundle_fundamental.joblib"]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise RuntimeError(f"Відсутні файли: {missing}. Запусти build_features, а потім train.")

    features = pd.read_parquet(FEATURES_PATH)
    features["match_date"] = pd.to_datetime(features["match_date"])
    ART["features"] = features.reset_index(drop=True)
    ART["by_id"] = {int(mid): i for i, mid in enumerate(ART["features"]["match_id"])}
    ART["form"] = _build_form(ART["features"])
    ART["state"] = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    ART["metrics"] = json.loads((MODELS_DIR / "metrics.json").read_text(encoding="utf-8"))
    ART["bundles"] = {k: joblib.load(MODELS_DIR / f"bundle_{k}.joblib") for k in ("market", "fundamental")}
    yield
    ART.clear()


app = FastAPI(title="EPL Match Predictor API", version="3.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Допоміжні функції ----------
def _num(x) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


def _form(team: str, before=None) -> list[str]:
    items = ART["form"].get(team, [])
    if before is not None:
        items = [x for x in items if x[0] < before]
    return [c for _, c in items[-5:]]


def _context(name: str, r: pd.Series, side: str) -> TeamContext:
    return TeamContext(name=name, elo=float(r[f"elo_{side}"]), xgf_5=float(r[f"{side}_xgf_5"]),
                       xga_5=float(r[f"{side}_xga_5"]), pts_5=float(r[f"{side}_pts_5"]), crest=fpl.crest(name))


def _explain(bundle: dict, X: pd.DataFrame, cls: int, top_n: int = 8) -> list[ShapFactor]:
    try:
        sv = bundle["cb"].get_feature_importance(Pool(X), type="ShapValues")
        contrib = sv[0, cls, :-1]
    except Exception:
        return []
    order = np.argsort(-np.abs(contrib))[:top_n]
    return [ShapFactor(feature=bundle["cols"][i], value=_num(X.iloc[0, i]), impact=float(contrib[i])) for i in order]


def _align(matrix: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Форма розподілу рахунків лишається від Dixon-Coles, а сумарні П1/X/П2 вирівнюються під ML-модель."""
    gh, ga = np.indices(matrix.shape)
    out = matrix.copy()
    for mask, p in ((gh > ga, probs[0]), (gh == ga, probs[1]), (gh < ga, probs[2])):
        s = out[mask].sum()
        if s > 0:
            out[mask] *= p / s
    return out


def _predict(row: pd.DataFrame, home: str, away: str, *, use_market: bool, match_id: int | None = None,
             actual: Actual | None = None, split: str | None = None, bookmaker: list[float] | None = None, base_probs: list[float] | None = None,
             explain: bool = True, adjustment: Adjustment | None = None, h2h=None,
             data_until: str | None = None) -> Prediction:
    key = "market" if use_market else "fundamental"
    bundle = ART["bundles"][key]
    X = row[bundle["cols"]]
    probs = bundle["ensemble"].predict_proba(X)[0]
    cls = int(np.argmax(probs))
    r = row.iloc[0]
    before = r["match_date"] if "match_date" in row.columns else None

    matrix = _align(predict_score_matrix(float(r["exp_home_goals"]), float(r["exp_away_goals"]))[3], probs)
    gh, ga = np.indices(matrix.shape)
    over_25 = float(matrix[gh + ga >= 3].sum())
    btts = float(matrix[(gh > 0) & (ga > 0)].sum())

    pred_code = CODES[cls]
    outcome_text = {"H": f"Перемога {home} (П1)", "D": "Нічия (X)", "A": f"Перемога {away} (П2)"}
    if actual is not None:
        actual.correct = actual.result == pred_code

    return Prediction(
        match_id=match_id, home_team=home, away_team=away,
        prob_home_win=float(probs[0]), prob_draw=float(probs[1]), prob_away_win=float(probs[2]),
        predicted_class=pred_code, predicted_outcome=outcome_text[pred_code], model_used=key,
        expected_home_goals=float(r["exp_home_goals"]), expected_away_goals=float(r["exp_away_goals"]),
        exact_scores=[ScoreProb(score=s, prob=p) for s, p in top_scores(matrix, 5)],
        score_matrix=[[round(float(v), 4) for v in line[:6]] for line in matrix[:6]],
        over_25=over_25, btts=btts,
        shap_factors=_explain(bundle, X, cls) if explain else [],
        home_context=_context(home, r, "home"), away_context=_context(away, r, "away"),
        home_form=_form(home, before), away_form=_form(away, before), h2h=h2h,
        bookmaker_probs=bookmaker, actual=actual, split=split, adjustment=adjustment, data_until=data_until, base_probs=base_probs,
    )


def _custom(home: str, away: str, adjust: bool = True, explain: bool = True) -> Prediction:
    state = ART["state"]
    hs, aw = dict(state[home]), dict(state[away])
    adjustment, base = None, None
    if adjust:
        sh, sa = fpl.squad(home), fpl.squad(away)
        if sh and sa:
            bundle = ART["bundles"]["fundamental"]
            base_row = matchup_from_state(hs, aw)
            base = [float(v) for v in bundle["ensemble"].predict_proba(base_row[bundle["cols"]])[0]]
            hs["elo"] += sh["elo_penalty"]
            aw["elo"] += sa["elo_penalty"]
            adjustment = Adjustment(home_elo=sh["elo_penalty"], away_elo=sa["elo_penalty"])
    row = matchup_from_state(hs, aw)
    row["home_team"], row["away_team"] = home, away
    return _predict(row, home, away, use_market=False, explain=explain, adjustment=adjustment,
                    h2h=state[home].get("h2h", {}).get(away), base_probs=base,
                    data_until=max(hs["last_match"], aw["last_match"]))


# ---------- Ендпоінти ----------
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/api/v1/health")
def health():
    boot = fpl.bootstrap()
    mapped = fpl.name_map(ART["state"].keys())
    missing = [t["name"] for t in boot["teams"] if t["id"] not in mapped] if boot else []
    return {"status": "ok", "matches": len(ART["features"]), "teams": len(ART["state"]),
            "fpl_available": boot is not None, "missing_teams": missing}


@app.get("/api/v1/teams", response_model=list[TeamOption])
def teams():
    allowed = set(fpl.name_map(ART["state"].keys()).values())        # поточні 20 клубів ∩ БД
    if not allowed:
        allowed = {n for n, s in ART["state"].items() if s["active"]}
    out = []
    for name in sorted(allowed):
        t = fpl.find_team(name)
        out.append(TeamOption(name=name, short=t["short_name"] if t else name[:3].upper(), crest=fpl.badge(t)))
    return out


@app.get("/api/v1/squad/{team}")
def squad(team: str):
    s = fpl.squad(team)
    if s is None:
        raise HTTPException(503, "Дані FPL недоступні або команду не знайдено")
    return s


@app.get("/api/v1/fixtures/next", response_model=NextRound)
def _round(event: int | None, adjust: bool) -> NextRound:
    rnd = fpl.upcoming_round(event)
    if rnd is None:
        raise HTTPException(503, "Календар FPL недоступний або такого туру немає")
    names = fpl.name_map(ART["state"].keys())
    out = []
    for f in rnd["fixtures"]:
        h, a = names.get(f["home_id"]), names.get(f["away_id"])
        item = FixtureOut(kickoff=f["kickoff"], home_team=h or f["home_name"], away_team=a or f["away_name"],
                          home_crest=f["home_crest"], away_crest=f["away_crest"])
        if h and a:
            try:
                p = _custom(h, a, adjust, explain=False)
                item.prob_home_win, item.prob_draw, item.prob_away_win = p.prob_home_win, p.prob_draw, p.prob_away_win
                item.predicted_class = p.predicted_class
            except Exception:
                log.exception("fixture prediction failed: %s - %s", h, a)
                item.note = "Не вдалося обчислити прогноз"
        else:
            item.note = "Команди немає в БД: запустіть sync_football_data"
        if f["finished"] and f["score"]:
            item.note = f"Зіграно: {f['score']}"
        out.append(item)
    return NextRound(event=rnd["event"], fixtures=out)


@app.get("/api/v1/fixtures/next", response_model=NextRound)
def next_round(adjust: bool = True):
    return _round(None, adjust)


@app.get("/api/v1/fixtures/round/{event}", response_model=NextRound)
def round_by_event(event: int, adjust: bool = True):
    return _round(event, adjust)


@app.get("/api/v1/matches/recent", response_model=list[MatchOut])
def recent_matches(limit: int = Query(40, ge=1, le=200)):
    df = ART["features"].dropna(subset=["target"]).sort_values(["match_date", "match_id"], ascending=False).head(limit)
    return [MatchOut(id=int(r.match_id), season=str(r.season), match_date=r.match_date.strftime("%Y-%m-%d"),
                     home_team=r.home_team, away_team=r.away_team, home_score=int(r.home_score),
                     away_score=int(r.away_score), result_1x2=r.result_1x2, split=str(r.split))
            for r in df.itertuples()]


@app.post("/api/v1/predict/custom", response_model=Prediction)
def predict_custom(req: CustomRequest):
    if req.home_team == req.away_team:
        raise HTTPException(400, "Команди повинні бути різними")
    for name in (req.home_team, req.away_team):
        if name not in ART["state"]:
            raise HTTPException(404, f"Невідома команда: {name}")
    return _custom(req.home_team, req.away_team, req.adjust_absences)


@app.post("/api/v1/predict/{match_id}", response_model=Prediction)
def predict_match(match_id: int):
    idx = ART["by_id"].get(match_id)
    if idx is None:
        raise HTTPException(404, "Матч не знайдено")
    row = ART["features"].iloc[[idx]]
    r = row.iloc[0]
    shin = [float(r[c]) for c in SHIN_COLS]
    has_odds = all(math.isfinite(v) for v in shin)
    actual = Actual(score=f"{int(r.home_score)}:{int(r.away_score)}", result=r.result_1x2, correct=False)
    return _predict(row, r.home_team, r.away_team, use_market=has_odds, match_id=match_id, actual=actual,
                    split=str(r.split), bookmaker=shin if has_odds else None)


@app.get("/api/v1/analytics/metrics")
def metrics():
    return ART["metrics"]

from src.api.extra import router as extra_router  # noqa: E402

app.include_router(extra_router, prefix="/api/v1")