"""Навчання: добір набору ознак, CatBoost (для ринку: поправка до логітів) + LightGBM + LogReg,
рефіт на train+val, температурне калібрування, чесні бейзлайни."""
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features.schema import DC_COLS, FUND_SETS, MARKET_SETS, SHIN_COLS
from src.ml.ensemble import EnsembleModel, MarketOffsetCatBoost, fit_temperature, rps_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURES_PATH = Path("data/processed/epl_features.parquet")
MODELS_DIR = Path("models")
SEED, N_TRIALS, MIN_GAMES = 42, 20, 5
BASE_CB = dict(iterations=800, loss_function="MultiClass", eval_metric="MultiClass", random_seed=SEED,
               verbose=0, thread_count=-1, allow_writing_files=False)
LGB = dict(learning_rate=0.02, num_leaves=6, min_child_samples=50, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.6, reg_lambda=10.0, random_state=SEED, verbose=-1)


def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    return {"accuracy": round(accuracy_score(y, p.argmax(1)) * 100, 2),
            "log_loss": round(log_loss(y, p, labels=[0, 1, 2]), 4),
            "rps": round(rps_score(y, p), 4)}


def inner(m):
    return m.model if isinstance(m, MarketOffsetCatBoost) else m


def fit_cb(params: dict, offset: bool, Xtr, ytr, Xva=None, yva=None):
    if offset:
        kw = {"early_stopping_rounds": 40} if Xva is not None else {}
        return MarketOffsetCatBoost(**params).fit(Xtr, ytr, Xva, yva, **kw)
    model = CatBoostClassifier(**params)
    if Xva is None:
        return model.fit(Xtr, ytr)
    return model.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=40)


def fit_bundle(label: str, cols: list[str], tr, va, offset: bool) -> dict:
    Xtr, ytr, Xva, yva = tr[cols], tr["target"].astype(int), va[cols], va["target"].astype(int)
    print(f"\n=== {label}: {len(cols)} ознак, offset={offset} ===")

    def objective(trial: optuna.Trial) -> float:
        if offset:   # поправка до ринку має бути невеликою: мілкі дерева, сильна регуляризація
            space = {"depth": trial.suggest_int("depth", 2, 4),
                     "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
                     "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 5.0, 60.0, log=True)}
        else:
            space = {"depth": trial.suggest_int("depth", 3, 6),
                     "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                     "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True)}
        space["random_strength"] = trial.suggest_float("random_strength", 0.1, 3.0)
        space["rsm"] = trial.suggest_float("rsm", 0.4, 1.0)
        m = fit_cb({**BASE_CB, **space}, offset, Xtr, ytr, Xva, yva)
        return inner(m).get_best_score()["validation"]["MultiClass"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS)
    params = {**BASE_CB, **study.best_params}
    print(f"CatBoost val logloss={study.best_value:.4f}")

    # 1) моделі на train: для ваг, температури та вибору набору ознак
    cb = fit_cb(params, offset, Xtr, ytr, Xva, yva)
    lgbm = LGBMClassifier(n_estimators=800, **LGB).fit(
        Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[early_stopping(40, verbose=False)])
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    lr.fit(Xtr, ytr)
    val_probs = [m.predict_proba(Xva) for m in (cb, lgbm, lr)]

    best_w, best_loss = None, np.inf
    for i in range(11):
        for j in range(11 - i):
            w = np.array([i, j, 10 - i - j]) / 10.0
            loss = log_loss(yva, sum(wk * pk for wk, pk in zip(w, val_probs)), labels=[0, 1, 2])
            if loss < best_loss:
                best_w, best_loss = w, loss
    weights = 0.5 * best_w + 0.5 / 3.0               # зсув до рівних ваг, щоб ансамбль не схлопувався в одну модель

    probe = EnsembleModel([cb, lgbm, lr], weights)
    temperature = fit_temperature(probe.blend(Xva), yva.to_numpy())
    probe.temperature = temperature
    val_loss = log_loss(yva, probe.predict_proba(Xva), labels=[0, 1, 2])
    print(f"Ваги (CB, LGBM, LR)={np.round(weights, 3)}, T={temperature:.3f}, val logloss={val_loss:.4f}")

    # 2) фінальні моделі: рефіт на train+val з тією ж кількістю ітерацій (+15%)
    Xall, yall = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])
    cb_f = fit_cb({**params, "iterations": max(60, int((inner(cb).get_best_iteration() + 1) * 1.15))},
                  offset, Xall, yall)
    n_lgb = max(60, int((lgbm.best_iteration_ or 800) * 1.15))
    lgb_f = LGBMClassifier(n_estimators=n_lgb, **LGB).fit(Xall, yall)
    lr_f = clone(lr).fit(Xall, yall)

    ensemble = EnsembleModel([cb_f, lgb_f, lr_f], weights, temperature)
    return {"label": label, "cols": cols, "cb": inner(cb_f), "ensemble": ensemble,
            "val_loss": float(val_loss), "offset": offset}


def pick_best(sets: dict[str, list[str]], tr, va, offset: bool) -> tuple[dict, list[dict]]:
    best, rows = None, []
    for name, cols in sets.items():
        bundle = fit_bundle(name, cols, tr, va, offset)
        rows.append({"set": name, "n_features": len(cols), "val_log_loss": round(bundle["val_loss"], 4)})
        if best is None or bundle["val_loss"] < best["val_loss"]:
            best = {**bundle, "set": name}
    print(f"--> Обрано набір: {best['set']}")
    return best, rows


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Немає {FEATURES_PATH}. Спершу: python -m src.features.build_features")

    full = pd.read_parquet(FEATURES_PATH).drop(columns=["split"], errors="ignore")
    usable = full[(full.home_games_played >= MIN_GAMES) & (full.away_games_played >= MIN_GAMES)]
    n = len(usable)
    a, b = int(n * 0.75), int(n * 0.88)
    tr, va, te = usable.iloc[:a], usable.iloc[a:b], usable.iloc[b:]
    print(f"Train {len(tr)} | Val {len(va)} | Test {len(te)}  (лише хронологія)")

    full["split"] = "warmup"
    for name, part in (("train", tr), ("val", va), ("test", te)):
        full.loc[part.index, "split"] = name
    full.to_parquet(FEATURES_PATH, index=False)

    yte, ytr = te["target"].astype(int).to_numpy(), tr["target"].astype(int).to_numpy()
    MODELS_DIR.mkdir(exist_ok=True)
    results, ablation = [], {}

    def add(key: str, name: str, kind: str, probs: np.ndarray) -> None:
        results.append({"key": key, "name": name, "kind": kind, **evaluate(yte, probs)})

    prior = np.bincount(ytr, minlength=3) / len(ytr)
    add("prior", "Частоти результатів (без ознак)", "baseline", np.tile(prior, (len(te), 1)))
    elo_lr = LogisticRegression().fit(tr[["elo_diff"]], ytr)
    add("elo", "Elo + логістична регресія", "baseline", elo_lr.predict_proba(te[["elo_diff"]]))
    add("dc", "Dixon-Coles (xG-форма)", "baseline", te[DC_COLS].to_numpy())
    shin = te[SHIN_COLS].to_numpy().copy()
    shin[np.isnan(shin).any(axis=1)] = prior
    add("bookmaker", "Букмекери (Bet365, Shin)", "benchmark", shin)

    for key, label, sets, offset in (("fundamental", "Наш ансамбль (без коефіцієнтів)", FUND_SETS, False),
                                     ("market", "Наш ансамбль (з коефіцієнтами)", MARKET_SETS, True)):
        bundle, rows = pick_best(sets, tr, va, offset)
        ablation[key] = rows
        joblib.dump(bundle, MODELS_DIR / f"bundle_{key}.joblib")
        add(key, label, "ours", bundle["ensemble"].predict_proba(te[bundle["cols"]]))

    summary = {"train_samples": len(tr), "val_samples": len(va), "test_samples": len(te),
               "test_from": te["match_date"].min().strftime("%Y-%m-%d"),
               "test_to": te["match_date"].max().strftime("%Y-%m-%d"),
               "models": results, "ablation": ablation}
    (MODELS_DIR / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"{'Модель':<40}{'Accuracy':>10}{'LogLoss':>10}{'RPS':>10}")
    for r in results:
        print(f"{r['name']:<40}{r['accuracy']:>9.2f}%{r['log_loss']:>10.4f}{r['rps']:>10.4f}")
    print("=" * 78)


if __name__ == "__main__":
    main()