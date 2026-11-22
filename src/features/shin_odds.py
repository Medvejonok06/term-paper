"""Метод Шіна для видалення маржі букмекера з коефіцієнтів 1X2."""
import numpy as np
from scipy.optimize import brentq

NAN3 = (float("nan"), float("nan"), float("nan"))


def shin_de_margin(odds_h: float, odds_d: float, odds_a: float) -> tuple[float, float, float]:
    """Повертає (p_home, p_draw, p_away). Якщо коефіцієнти відсутні, повертає NaN."""
    odds = np.array([odds_h, odds_d, odds_a], dtype=float)
    if not np.all(np.isfinite(odds)) or np.any(odds <= 1.0):
        return NAN3

    q = 1.0 / odds            # «сирі» ймовірності з маржею
    total = q.sum()           # booksum > 1
    if total <= 1.0 + 1e-9:
        p = q / total
        return float(p[0]), float(p[1]), float(p[2])

    def probs(z: float) -> np.ndarray:
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * q * q / total) - z) / (2.0 * (1.0 - z))

    try:
        # f(0) = sqrt(total) - 1 > 0, а при z=0.5 сума вже менша за 1, тож корінь є
        z = brentq(lambda t: probs(t).sum() - 1.0, 0.0, 0.5)
        p = probs(z)
    except ValueError:
        p = q  # запасний варіант: проста нормалізація

    p = p / p.sum()
    return float(p[0]), float(p[1]), float(p[2])