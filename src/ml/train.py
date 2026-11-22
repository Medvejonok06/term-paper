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
import warnings
from sklearn.model_selection import TimeSeriesSplit
from src.ml.ensemble import (
    ColumnLR, EnsembleModel, MarketOffsetCatBoost, ShinOnly, apply_temperature, fit_temperature, rps_score,
)

warnings.filterwarnings("ignore", message=".*eval_set.*")
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


def weight_grid(k: int, step: int = 10) -> list[np.ndarray]:
    def rec(n, left):
        if n == 1:
            yield (left,)
            return
        for i in range(left + 1):
            for rest in rec(n - 1, left - i):
                yield (i, *rest)
    return [np.array(w) / step for w in rec(k, step)]

def drift_factors(probs: np.ndarray, y: np.ndarray, window: int, alpha: float) -> np.ndarray:
    """Множники для виправлення зсуву частот результатів за останніми `window` матчами."""
    p, o = probs[-window:], y[-window:]
    obs = np.clip(np.bincount(o, minlength=3) / len(o), 1e-3, 1.0)
    return (obs / p.mean(axis=0)) ** alpha


def sequential_adapt(probs: np.ndarray, y: np.ndarray, start: int, window: int, alpha: float) -> np.ndarray:
    """Кожен прогноз i >= start коригується лише за матчами ДО нього (без витоку)."""
    out = probs.copy()
    for i in range(start, len(probs)):
        q = probs[i] * drift_factors(probs[:i], y[:i], window, alpha)
        out[i] = q / q.sum()
    return out


def choose_drift(oof: np.ndarray, y: np.ndarray, warmup: int = 150,
                 windows=(150, 300, 500), alphas=(0.5, 1.0)) -> dict | None:
    def loss(p):
        return log_loss(y[warmup:], p[warmup:], labels=[0, 1, 2])

    base, best = loss(oof), None
    for w in windows:
        for a in alphas:
            l = loss(sequential_adapt(oof, y, warmup, w, a))
            if l < base - 0.001 and (best is None or l < best["oof_loss"]):
                best = {"window": w, "alpha": a, "oof_loss": round(float(l), 4), "oof_base": round(float(base), 4)}
    print(f"Корекція дрейфу: {best if best else 'не покращила OOF, не застосовую'} (база {base:.4f})")
    return best


def oof_predict(factories: dict, X, y, n_splits: int = 4, test_size: int = 380):
    """Walk-forward: кожен фолд вчиться на минулому й прогнозує наступні ~380 матчів (сезон)."""
    idx, preds = [], {k: [] for k in factories}
    for tr_i, te_i in TimeSeriesSplit(n_splits=n_splits, test_size=test_size).split(X):
        for k, make in factories.items():
            m = make().fit(X.iloc[tr_i], y.iloc[tr_i])
            preds[k].append(m.predict_proba(X.iloc[te_i]))
        idx.append(te_i)
    return np.concatenate(idx), {k: np.vstack(v) for k, v in preds.items()}


def fit_bundle(label: str, cols: list[str], tr, va, offset: bool) -> dict:
    Xtr, ytr, Xva, yva = tr[cols], tr["target"].astype(int), va[cols], va["target"].astype(int)
    print(f"\n=== {label}: {len(cols)} ознак, offset={offset} ===")

    def objective(trial: optuna.Trial) -> float:
        if offset:
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

    # кількість ітерацій з ранньою зупинкою на val, +15% для більшої вибірки
    probe = fit_cb(params, offset, Xtr, ytr, Xva, yva)
    n_cb = max(60, int((inner(probe).get_best_iteration() + 1) * 1.15))
    n_lgb = 300
    if not offset:
        lg = LGBMClassifier(n_estimators=800, **LGB).fit(
            Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[early_stopping(40, verbose=False)])
        n_lgb = max(60, int((lg.best_iteration_ or 800) * 1.15))

    cb_params = {**params, "iterations": n_cb}
    if offset:
        factories = {"market": ShinOnly, "cb_offset": lambda: MarketOffsetCatBoost(**cb_params)}
    else:
        factories = {
            "elo_lr": lambda: ColumnLR(["elo_diff"]),
            "cb": lambda: CatBoostClassifier(**cb_params),
            "lgbm": lambda: LGBMClassifier(n_estimators=n_lgb, **LGB),
            "lr_all": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                            LogisticRegression(C=0.1, max_iter=3000)),
        }

    Xall, yall = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])
    idx, oof = oof_predict(factories, Xall, yall)
    y_oof = yall.iloc[idx].to_numpy()
    names = list(factories)

    best_w, best_loss = None, np.inf
    for w in weight_grid(len(names)):
        loss = log_loss(y_oof, sum(wk * oof[n] for wk, n in zip(w, names)), labels=[0, 1, 2])
        if loss < best_loss:
            best_w, best_loss = w, loss
    blend = sum(wk * oof[n] for wk, n in zip(best_w, names))
    temperature = fit_temperature(blend, y_oof)
    oof_loss = log_loss(y_oof, apply_temperature(blend, temperature), labels=[0, 1, 2])
    members = {n: round(log_loss(y_oof, oof[n], labels=[0, 1, 2]), 4) for n in names}
    print(f"OOF ({len(y_oof)} матчів), logloss членів: {members}")
    print(f"Ваги {dict(zip(names, best_w.round(2).tolist()))}, T={temperature:.3f}, OOF logloss={oof_loss:.4f}")

    oof_probs = apply_temperature(blend, temperature)
    fitted = {n: make().fit(Xall, yall) for n, make in factories.items()}       # рефіт на train+val
    ensemble = EnsembleModel(list(fitted.values()), best_w, temperature)
    cb_model = fitted["cb"] if "cb" in fitted else fitted["cb_offset"].model
    return {"label": label, "cols": cols, "cb": cb_model, "ensemble": ensemble, "val_loss": float(oof_loss),
            "offset": offset, "weights": dict(zip(names, best_w.tolist())),
            "oof_probs": oof_probs, "y_oof": y_oof}
    return {"label": label, "cols": cols, "cb": cb_model, "ensemble": ensemble, "val_loss": float(oof_loss),
            "offset": offset, "weights": dict(zip(names, best_w.tolist()))}


def pick_best(sets: dict[str, list[str]], tr, va, offset: bool) -> tuple[dict, list[dict]]:
    best, rows = None, []
    for name, cols in sets.items():
        bundle = fit_bundle(name, cols, tr, va, offset)
        rows.append({"set": name, "n_features": len(cols), "oof_log_loss": round(bundle["val_loss"], 4),
                     "weights": bundle["weights"]})
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
        p_test = bundle["ensemble"].predict_proba(te[bundle["cols"]])
        add(key, label, "ours", p_test)

        oof, y_oof = bundle.pop("oof_probs"), bundle.pop("y_oof")
        cfg = choose_drift(oof, y_oof)
        if cfg:
            all_p, all_y = np.vstack([oof, p_test]), np.concatenate([y_oof, yte])
            adapted = sequential_adapt(all_p, all_y, len(oof), cfg["window"], cfg["alpha"])[len(oof):]
            add(f"{key}_adapt", f"{label} + адаптація до дрейфу", "ours", adapted)
            bundle["drift"] = {**cfg, "factors": drift_factors(all_p, all_y, cfg["window"], cfg["alpha"]).tolist()}
        joblib.dump(bundle, MODELS_DIR / f"bundle_{key}.joblib")

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