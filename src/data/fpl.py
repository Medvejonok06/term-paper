"""Клієнт Fantasy Premier League API (безкоштовний, без ключа): емблеми, склади, травми, календар.
API неофіційний: структура може змінитися, тому є кеш і безпечні запасні варіанти."""
from __future__ import annotations

import difflib
import json
import re
import time
from pathlib import Path

import requests

BASE = "https://fantasy.premierleague.com/api/"
BADGE = "https://resources.premierleague.com/premierleague/badges/50/t{code}.png"
PHOTO = "https://resources.premierleague.com/premierleague25/photos/players/110x140/{code}.png"
CACHE_DIR = Path("data/cache")
TTL = 1800
HEADERS = {"User-Agent": "epl-predictor-coursework/1.0"}
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# назва (без апострофів, нижній регістр) -> назва, як у football-data.co.uk (ваша БД)
ALIASES = {
    "man utd": "man united", "manchester united": "man united", "manchester city": "man city",
    "spurs": "tottenham", "tottenham hotspur": "tottenham", "nottingham forest": "nottm forest",
    "newcastle united": "newcastle", "wolverhampton wanderers": "wolves", "leicester city": "leicester",
    "leeds united": "leeds", "west ham united": "west ham", "brighton and hove albion": "brighton",
    "ipswich town": "ipswich", "sheffield utd": "sheffield united", "west bromwich albion": "west brom",
    "luton town": "luton", "norwich city": "norwich",
}

ALIASES.update({"hull city": "hull", "coventry city": "coventry"})

_MEM: dict[str, tuple[float, object]] = {}


def canon(name: str) -> str:
    s = re.sub(r"[’'`]", "", name.lower()).strip()
    return ALIASES.get(s, s)


def _get(path: str, check):
    """GET з кешем у пам'яті та на диску. Якщо мережа лягла, віддаємо застаріле."""
    now = time.time()
    hit = _MEM.get(path)
    if hit and now - hit[0] < TTL:
        return hit[1]

    f = CACHE_DIR / (re.sub(r"\W+", "_", path).strip("_") + ".json")
    try:
        r = requests.get(BASE + path, headers=HEADERS, timeout=8)
        r.raise_for_status()
        data = r.json()
        if check(data):
            _MEM[path] = (now, data)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(data), encoding="utf-8")
            return data
    except (requests.RequestException, ValueError):
        pass

    stale = hit[1] if hit else None
    if stale is None and f.exists():
        try:
            stale = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            stale = None
    if stale is not None:
        _MEM[path] = (now - TTL + 120, stale)      # наступна спроба мережі через 2 хв
    return stale


def bootstrap() -> dict | None:
    return _get("bootstrap-static/", lambda d: isinstance(d, dict) and "elements" in d and "teams" in d)


def fixtures() -> list | None:
    return _get("fixtures/", lambda d: isinstance(d, list))


def find_team(name: str) -> dict | None:
    boot = bootstrap()
    if not boot:
        return None
    index = {canon(t["name"]): t for t in boot["teams"]}
    key = canon(name)
    if key in index:
        return index[key]
    close = difflib.get_close_matches(key, list(index), n=1, cutoff=0.8)
    return index[close[0]] if close else None


def badge(team: dict | None) -> str | None:
    return BADGE.format(code=team["code"]) if team else None


def crest(name: str) -> str | None:
    return badge(find_team(name))


def name_map(our_names) -> dict[int, str]:
    """id команди у FPL -> назва в нашій БД."""
    boot = bootstrap()
    if not boot:
        return {}
    ours = {canon(n): n for n in our_names}
    out = {}
    for t in boot["teams"]:
        key = canon(t["name"])
        if key in ours:
            out[t["id"]] = ours[key]
        else:
            close = difflib.get_close_matches(key, list(ours), n=1, cutoff=0.8)
            if close:
                out[t["id"]] = ours[close[0]]
    return out


def games_played(team_id: int) -> int:
    return sum(1 for f in (fixtures() or [])
               if (f.get("finished") or f.get("finished_provisional")) and team_id in (f["team_h"], f["team_a"]))


def upcoming_round(event: int | None = None) -> dict | None:
    fx, boot = fixtures(), bootstrap()
    if not fx or not boot:
        return None

    def done(f):
        return bool(f.get("finished") or f.get("finished_provisional"))

    if event is None:
        todo = [f for f in fx if f.get("event") and not done(f)]
        if not todo:
            return None
        event = min(f["event"] for f in todo)
        pool = [f for f in todo if f["event"] == event]
    else:
        pool = [f for f in fx if f.get("event") == event]
        if not pool:
            return None

    teams = {t["id"]: t for t in boot["teams"]}
    rows = []
    for f in sorted(pool, key=lambda f: f.get("kickoff_time") or ""):
        h, a = teams[f["team_h"]], teams[f["team_a"]]
        has_score = done(f) and f.get("team_h_score") is not None
        rows.append({"kickoff": f.get("kickoff_time"), "home_id": h["id"], "away_id": a["id"],
                     "home_name": h["name"], "away_name": a["name"],
                     "home_crest": badge(h), "away_crest": badge(a), "finished": done(f),
                     "score": f"{f['team_h_score']}:{f['team_a_score']}" if has_score else None})
    return {"event": event, "fixtures": rows}


# ---------- склади ----------
def _availability(p: dict) -> float:
    """0 = не зіграє, 1 = грає. Прогноз лікарів ≥75% вважаємо повною доступністю."""
    status, chance = p["status"], p.get("chance_of_playing_next_round")
    if status in ("u", "n", "s"):                 # пішов, недоступний, дискваліфікований
        return 0.0
    if chance is not None:
        return 1.0 if chance >= 75 else chance / 100.0
    return 0.0 if status == "i" else 0.5 if status == "d" else 1.0


def _view(p: dict) -> dict:
    code = str(p.get("photo") or "").split(".")[0]
    return {"id": p["id"], "name": p["web_name"], "pos": POS.get(p["element_type"], "MID"),
            "photo": PHOTO.format(code=code) if code else None,
            "status": p["status"], "chance": p.get("chance_of_playing_next_round"),
            "news": p.get("news") or "", "yellow": p.get("yellow_cards", 0), "red": p.get("red_cards", 0),
            "minutes": p.get("minutes", 0)}


def _pick_xi(pool: list[dict]) -> list[dict]:
    """1 воротар + найсильніші польові з обмеженнями схеми (3-5 захисників, 2-5 півзахисників, 1-3 нападники)."""
    def rank(p):
        return p["score"] * p["availability"]

    by = {pos: sorted((p for p in pool if p["pos"] == pos), key=lambda p: -rank(p)) for pos in POS.values()}
    need, cap = {"DEF": 3, "MID": 2, "FWD": 1}, {"DEF": 5, "MID": 5, "FWD": 3}
    chosen = {pos: by[pos][:n] for pos, n in need.items()}
    rest = sorted((p for pos in need for p in by[pos][need[pos]:]), key=lambda p: -rank(p))
    left = 10 - sum(len(v) for v in chosen.values())
    for p in rest:
        if left <= 0:
            break
        if len(chosen[p["pos"]]) < cap[p["pos"]]:
            chosen[p["pos"]].append(p)
            left -= 1
    return by["GK"][:1] + chosen["DEF"] + chosen["MID"] + chosen["FWD"]


def _ban_window(played: int) -> tuple[int, int]:
    """(поріг жовтих, скільки матчів ще діє вікно). 5 до 19-го матчу, 10 до 32-го, 15 до кінця."""
    nxt = played + 1
    if nxt <= 19:
        return 5, 19 - played
    if nxt <= 32:
        return 10, 32 - played
    return 15, 38 - played


def squad(team_name: str) -> dict | None:
    boot, team = bootstrap(), find_team(team_name)
    if not boot or not team:
        return None
    raw = [e for e in boot["elements"] if e["team"] == team["id"]]
    if not raw:
        return None

    top_min = max(p.get("minutes", 0) for p in raw)
    top_cost = max(p["now_cost"] for p in raw) or 1
    w = min(1.0, top_min / 900.0)
    players = []
    for p in raw:
        share = p.get("minutes", 0) / top_min if top_min else 0.0
        score = 0.8 * w * share + (1 - 0.8 * w) * p["now_cost"] / top_cost
        v = _view(p)
        v.update(score=round(score, 3), availability=_availability(p),
                 role="core" if score >= 0.6 else "rotation" if score >= 0.3 else "squad")
        players.append(v)

    by_score = sorted(players, key=lambda p: -p["score"])
    xi = _pick_xi([p for p in players if p["availability"] >= 0.5])
    in_xi = {p["id"] for p in xi}
    rest = sorted((p for p in players if p["availability"] > 0 and p["id"] not in in_xi),
                  key=lambda p: -p["score"] * p["availability"])
    gk2 = next((p for p in rest if p["pos"] == "GK"), None)
    bench = ([gk2] if gk2 else []) + [p for p in rest if p is not gk2][:6]

    absent = [p for p in by_score if p["availability"] < 0.5 and p["status"] != "u"
              and (p["score"] >= 0.12 or p["status"] == "s")][:8]
    doubtful = [p for p in by_score if 0.5 <= p["availability"] < 1.0][:6]
    likely_back = [p for p in by_score if p["status"] in ("i", "d") and p["availability"] == 1.0][:5]
    left = [p for p in by_score if p["status"] == "u" and p["score"] >= 0.12][:5]

    def wt(p):
        return p["score"] * (1.3 if p["pos"] == "GK" else 1.0)

    key14 = by_score[:14]
    missing = sum(wt(p) * (1 - p["availability"]) for p in key14) / (sum(wt(p) for p in key14) or 1.0)
    elo_penalty = -round(min(80.0, 160.0 * missing), 1)

    played = games_played(team["id"])
    limit, left_games = _ban_window(played)
    ban_risk = [{"id": p["id"], "name": p["name"], "yellow": p["yellow"], "limit": limit, "matches_left": left_games}
                for p in by_score if p["yellow"] == limit - 1 and p["status"] != "u"][:5]

    return {"team": team_name, "crest": badge(team),
            "formation": f"{sum(p['pos'] == 'DEF' for p in xi)}-{sum(p['pos'] == 'MID' for p in xi)}-"
                         f"{sum(p['pos'] == 'FWD' for p in xi)}",
            "starting_xi": xi, "bench": bench, "absent": absent, "doubtful": doubtful, "likely_back": likely_back,
            "left": left, "ban_risk": ban_risk, "missing_strength": round(missing, 3), "elo_penalty": elo_penalty,
            "games_played": played, "source": "Fantasy Premier League",
            "updated": time.strftime("%Y-%m-%d %H:%M", time.localtime(_MEM.get("bootstrap-static/", (time.time(),))[0]))}

def current_names() -> list[str]:
    boot = bootstrap()
    return [t["name"] for t in boot["teams"]] if boot else []


def remaining() -> list[tuple[int, int]]:
    return [(f["team_h"], f["team_a"]) for f in (fixtures() or [])
            if not (f.get("finished") or f.get("finished_provisional"))]


def table() -> list[dict]:
    """Поточна турнірна таблиця за зіграними матчами календаря FPL."""
    boot, fx = bootstrap(), fixtures()
    if not boot or fx is None:
        return []
    rows = {t["id"]: {"id": t["id"], "name": t["name"], "crest": badge(t), "P": 0, "W": 0, "D": 0, "L": 0,
                      "GF": 0, "GA": 0, "Pts": 0, "form": []} for t in boot["teams"]}
    played = [f for f in fx if (f.get("finished") or f.get("finished_provisional"))
              and f.get("team_h_score") is not None]
    for f in sorted(played, key=lambda f: f.get("kickoff_time") or ""):
        for me, gf, ga in ((f["team_h"], f["team_h_score"], f["team_a_score"]),
                           (f["team_a"], f["team_a_score"], f["team_h_score"])):
            r = rows[me]
            res = "W" if gf > ga else "D" if gf == ga else "L"
            r["P"] += 1; r["GF"] += gf; r["GA"] += ga; r[res] += 1
            r["Pts"] += {"W": 3, "D": 1, "L": 0}[res]
            r["form"].append(res)
    out = sorted(rows.values(), key=lambda r: (-r["Pts"], -(r["GF"] - r["GA"]), -r["GF"], r["name"]))
    for i, r in enumerate(out, 1):
        r["pos"], r["GD"], r["form"] = i, r["GF"] - r["GA"], r["form"][-5:]
    return out