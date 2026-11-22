"""Додаткові ендпоінти: таблиця, Elo, профіль команди, H2H, травми, симуляція сезону, калібрування, важливість ознак."""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from src.data import fpl
from src.features.dixon_coles import expected_goals

router = APIRouter()


def _art() -> dict:
    from src.api.main import ART          # ледачий імпорт: уникаємо циклу main <-> extra
    return ART


@router.get("/standings")
def standings():
    art = _art()
    rows = fpl.table()
    if not rows:
        raise HTTPException(503, "Таблиця FPL недоступна")
    names = fpl.name_map(art["state"].keys())
    for r in rows:
        our = names.get(r["id"])
        st = art["state"].get(our) if our else None
        r["team"] = our or r["name"]
        r["elo"] = round(st["elo"]) if st else None
    return rows


@router.get("/elo")
def elo_ranking(current_only: bool = True):
    state = _art()["state"]
    allowed = set(fpl.name_map(state.keys()).values()) if current_only else set(state)
    if not allowed:
        allowed = {n for n, s in state.items() if s["active"]}
    out = [{"team": n, "elo": round(s["elo"], 1), "crest": fpl.crest(n), "last_match": s["last_match"]}
           for n, s in state.items() if n in allowed]
    return sorted(out, key=lambda r: -r["elo"])


@router.get("/teams/{name}")
def team_profile(name: str):
    from src.api.main import _form
    art = _art()
    s = art["state"].get(name)
    if not s:
        raise HTTPException(404, f"Невідома команда: {name}")
    df = art["features"]
    last = df[(df.home_team == name) | (df.away_team == name)].tail(5)
    sq = fpl.squad(name)
    return {"team": name, "crest": fpl.crest(name), "elo": s["elo"], "last_match": s["last_match"], "form": _form(name),
            "xg_for_5": s["xgf_5"], "xg_against_5": s["xga_5"], "points_form_5": s["pts_5"],
            "at_home": s["venue_home"], "away": s["venue_away"], "table": s["table"],
            "recent_matches": [{"date": r.match_date.strftime("%Y-%m-%d"), "home": r.home_team, "away": r.away_team,
                                "score": f"{int(r.home_score)}:{int(r.away_score)}"} for r in last.itertuples()],
            "missing_strength": sq["missing_strength"] if sq else None,
            "absent": [p["name"] for p in sq["absent"]] if sq else []}


@router.get("/h2h")
def h2h(home: str, away: str, limit: int = Query(10, ge=1, le=50)):
    df = _art()["features"]
    m = df[((df.home_team == home) & (df.away_team == away)) | ((df.home_team == away) & (df.away_team == home))]
    m = m.sort_values("match_date", ascending=False).head(limit)
    if m.empty:
        raise HTTPException(404, "Очних зустрічей у БД немає")
    w = d = l = gf = ga = 0
    games = []
    for r in m.itertuples():
        hg, ag = (r.home_score, r.away_score) if r.home_team == home else (r.away_score, r.home_score)
        w, d, l = w + (hg > ag), d + (hg == ag), l + (hg < ag)
        gf, ga = gf + hg, ga + ag
        games.append({"date": r.match_date.strftime("%Y-%m-%d"), "home": r.home_team, "away": r.away_team,
                      "score": f"{int(r.home_score)}:{int(r.away_score)}"})
    return {"team": home, "opponent": away, "wins": int(w), "draws": int(d), "losses": int(l),
            "goals_for": int(gf), "goals_against": int(ga), "matches": games}


@router.get("/injuries")
def injuries():
    out = []
    for name in fpl.current_names():
        sq = fpl.squad(name)
        if sq:
            out.append({k: sq[k] for k in ("crest", "missing_strength", "elo_penalty", "absent", "doubtful",
                                           "ban_risk", "likely_back")} | {"team": name})
    if not out:
        raise HTTPException(503, "Дані FPL недоступні")
    return sorted(out, key=lambda r: -r["missing_strength"])


@router.get("/simulate/season")
def simulate(n: int = Query(3000, ge=500, le=20000)):
    """Монте-Карло решти сезону: Пуассон за xG-формою (Dixon-Coles без поправки, без травм)."""
    state = _art()["state"]
    rows, rem = fpl.table(), fpl.remaining()
    if not rows:
        raise HTTPException(503, "Календар FPL недоступний")
    names = fpl.name_map(state.keys())
    ids = [r["id"] for r in rows]
    ix = {t: i for i, t in enumerate(ids)}

    def strength(tid):
        s = state.get(names.get(tid, ""))
        return (s["xgf_10"], s["xga_10"]) if s else (1.1, 1.55)       # новачок без даних: обережне припущення

    rng = np.random.default_rng(42)
    pts = np.tile([r["Pts"] for r in rows], (n, 1)).astype(float)
    gf = np.tile([r["GF"] for r in rows], (n, 1)).astype(float)
    ga = np.tile([r["GA"] for r in rows], (n, 1)).astype(float)
    for h, a in rem:
        (ha, hc), (aa, ac) = strength(h), strength(a)
        lam, mu = expected_goals(ha, hc, aa, ac)
        gh, gg = rng.poisson(lam, n), rng.poisson(mu, n)
        hi, ai = ix[h], ix[a]
        pts[:, hi] += 3 * (gh > gg) + (gh == gg)
        pts[:, ai] += 3 * (gg > gh) + (gh == gg)
        gf[:, hi] += gh; ga[:, hi] += gg; gf[:, ai] += gg; ga[:, ai] += gh

    key = pts * 1e6 + (gf - ga) * 1e3 + gf + rng.random(pts.shape) * 1e-3
    order = np.argsort(-key, axis=1)
    pos = np.empty_like(order)
    np.put_along_axis(pos, order, np.broadcast_to(np.arange(len(ids)), order.shape), axis=1)

    teams = []
    for i, r in enumerate(rows):
        teams.append({"team": names.get(r["id"], r["name"]), "crest": r["crest"], "points_now": r["Pts"],
                      "exp_points": round(float(pts[:, i].mean()), 1), "exp_position": round(float(pos[:, i].mean()) + 1, 1),
                      "title": round(float((pos[:, i] == 0).mean()), 4), "top4": round(float((pos[:, i] < 4).mean()), 4),
                      "relegation": round(float((pos[:, i] >= len(ids) - 3).mean()), 4),
                      "assumed": r["id"] not in names})
    return {"simulations": n, "remaining_matches": len(rem), "teams": sorted(teams, key=lambda t: t["exp_position"])}


@router.get("/analytics/calibration")
def calibration(model: str = "market", bins: int = Query(8, ge=4, le=12)):
    art = _art()
    b = art["bundles"].get(model)
    if not b:
        raise HTTPException(404, "Невідома модель")
    cache = f"_calib_{model}_{bins}"
    if cache not in art:
        df = art["features"]
        te = df[df["split"] == "test"]
        p = b["ensemble"].predict_proba(te[b["cols"]])
        y = te["target"].astype(int).to_numpy()
        classes, ece = {}, 0.0
        for c, key in enumerate(("H", "D", "A")):
            pred, obs = p[:, c], (y == c).astype(float)
            edges = np.unique(np.quantile(pred, np.linspace(0, 1, bins + 1)))
            idx = np.clip(np.digitize(pred, edges[1:-1]), 0, len(edges) - 2)
            pts = []
            for k in range(len(edges) - 1):
                mask = idx == k
                if mask.any():
                    pts.append({"pred": round(float(pred[mask].mean()), 4), "obs": round(float(obs[mask].mean()), 4),
                                "n": int(mask.sum())})
                    ece += mask.sum() / len(pred) * abs(pred[mask].mean() - obs[mask].mean()) / 3
            classes[key] = pts
        art[cache] = {"n": int(len(te)), "ece": round(float(ece), 4), "classes": classes}
    return art[cache]


@router.get("/analytics/importance")
def importance(model: str = "fundamental", top: int = Query(15, ge=3, le=40)):
    b = _art()["bundles"].get(model)
    if not b:
        raise HTTPException(404, "Невідома модель")
    imp = b["cb"].get_feature_importance()
    return [{"feature": b["cols"][i], "importance": round(float(imp[i]), 3)} for i in np.argsort(-imp)[:top]]