"""Модель Діксона-Коулза: матриця рахунків та очікувані голи (рівень калібрується під дані)."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import poisson

CALIB_PATH = Path("data/processed/goal_calib.json")
SHRINK = 0.5                       # <-- залиште своє підібране значення
LEAGUE_AVG_GOALS, HOME_GOAL_FACTOR = 1.35, 1.12     # лишено для сумісності імпортів
# xg_mean: середній xG команди в ЦЬОМУ датасеті; home/away_goals: середні фактичні голи
CALIB = {"xg_mean": 1.35, "home_goals": 1.50, "away_goals": 1.20}
if CALIB_PATH.exists():
    CALIB.update(json.loads(CALIB_PATH.read_text()))


def set_calibration(xg_mean: float, home_goals: float, away_goals: float, save: bool = True) -> None:
    CALIB.update(xg_mean=float(xg_mean), home_goals=float(home_goals), away_goals=float(away_goals))
    if save:
        CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
        CALIB_PATH.write_text(json.dumps(CALIB))


def expected_goals(home_att, home_conc, away_att, away_conc):
    """Модель Махера зі стисненням до середнього й калібруванням рівня під реальні голи."""
    m = CALIB["xg_mean"]
    ha, hc = m + SHRINK * (home_att - m), m + SHRINK * (home_conc - m)
    aa, ac = m + SHRINK * (away_att - m), m + SHRINK * (away_conc - m)
    lam = ha * ac / m * (CALIB["home_goals"] / m)
    mu = aa * hc / m * (CALIB["away_goals"] / m)
    return np.clip(lam, 0.3, 3.2), np.clip(mu, 0.3, 3.0)


def match_lambdas(probs, total: float, rho: float = -0.05) -> tuple[float, float]:
    """Розкладає тотал голів на (λ, μ) так, щоб П1/П2 матриці Діксона-Коулза збіглися з ML-прогнозом."""
    best, best_err = (total / 2, total / 2), float("inf")
    for share in np.linspace(0.30, 0.80, 51):
        lam, mu = total * share, total * (1 - share)
        ph, _, pa, _ = predict_score_matrix(lam, mu, rho)
        err = (ph - probs[0]) ** 2 + (pa - probs[2]) ** 2
        if err < best_err:
            best, best_err = (lam, mu), err
    return best


def predict_score_matrix(lam: float, mu: float, rho: float = -0.05, max_goals: int = 8):
    """Повертає (P(home), P(draw), P(away), матриця[home_goals, away_goals])."""
    goals = np.arange(max_goals + 1)
    matrix = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))

    # Корекція Діксона-Коулза для низьких рахунків
    matrix[0, 0] *= 1.0 - lam * mu * rho
    matrix[0, 1] *= 1.0 + lam * rho
    matrix[1, 0] *= 1.0 + mu * rho
    matrix[1, 1] *= 1.0 - rho

    matrix = np.clip(matrix, 0.0, None)
    matrix /= matrix.sum()
    return (float(np.tril(matrix, -1).sum()), float(np.trace(matrix)),
            float(np.triu(matrix, 1).sum()), matrix)


def top_scores(matrix: np.ndarray, n: int = 5) -> list[tuple[str, float]]:
    flat = [(f"{h}:{a}", float(matrix[h, a])) for h in range(matrix.shape[0]) for a in range(matrix.shape[1])]
    return sorted(flat, key=lambda item: item[1], reverse=True)[:n]