"""Модель Діксона-Коулза: матриця ймовірностей рахунків та очікувані голи."""
import numpy as np
from scipy.stats import poisson

LEAGUE_AVG_GOALS = 1.35   # середня кількість голів команди за матч в EPL
HOME_GOAL_FACTOR = 1.12   # домашня перевага (~1.5 проти ~1.2 голу)


def expected_goals(home_att, home_conc, away_att, away_conc):
    """Мультиплікативна модель Махера. Працює і зі скалярами, і з pandas.Series."""
    lam = home_att * away_conc / LEAGUE_AVG_GOALS * HOME_GOAL_FACTOR
    mu = away_att * home_conc / LEAGUE_AVG_GOALS / HOME_GOAL_FACTOR
    return np.clip(lam, 0.2, 4.0), np.clip(mu, 0.2, 4.0)


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