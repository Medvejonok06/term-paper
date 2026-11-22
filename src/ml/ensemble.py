"""Ансамбль, температурне калібрування, RPS і CatBoost із поправкою до ринку."""
import numpy as np
from catboost import CatBoostClassifier, Pool
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

from src.features.schema import SHIN_COLS


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    return softmax(np.log(np.clip(probs, 1e-12, 1.0)) / t)


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    res = minimize_scalar(lambda t: log_loss(y, apply_temperature(probs, t), labels=[0, 1, 2]),
                          bounds=(0.5, 3.0), method="bounded")
    return float(res.x)


def rps_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[np.asarray(y_true, dtype=int)]
    diff = np.cumsum(probs, axis=1)[:, :-1] - np.cumsum(onehot, axis=1)[:, :-1]
    return float(np.mean(np.sum(diff ** 2, axis=1) / (probs.shape[1] - 1)))


class MarketOffsetCatBoost:
    """CatBoost вчиться не з нуля, а як поправка до логітів ринку (Shin): p = softmax(log p_market + f(x)).
    На малих вибірках це стабільніше: модель вчить лише те, чого ринок не знає."""

    def __init__(self, **params):
        self.params = params
        self.model: CatBoostClassifier | None = None

    @staticmethod
    def offset(X) -> np.ndarray:
        p = X[SHIN_COLS].to_numpy(dtype=float).copy()
        p[~np.isfinite(p).all(axis=1)] = [0.45, 0.26, 0.29]
        p = np.clip(p, 1e-4, 1.0)
        return np.log(p / p.sum(axis=1, keepdims=True))

    def fit(self, X, y, Xv=None, yv=None, **kw):
        self.model = CatBoostClassifier(**self.params)
        train = Pool(X, y, baseline=self.offset(X))
        valid = Pool(Xv, yv, baseline=self.offset(Xv)) if Xv is not None else None
        self.model.fit(train, eval_set=valid, **kw)
        return self

    def predict_proba(self, X) -> np.ndarray:
        raw = self.model.predict(X, prediction_type="RawFormulaVal")
        return softmax(self.offset(X) + raw)


class EnsembleModel:
    def __init__(self, models: list, weights: np.ndarray, temperature: float = 1.0):
        self.models = models
        self.weights = np.asarray(weights, dtype=float)
        self.temperature = temperature

    def blend(self, X) -> np.ndarray:
        return sum(w * m.predict_proba(X) for w, m in zip(self.weights, self.models))

    def predict_proba(self, X) -> np.ndarray:
        probs = self.blend(X)
        return apply_temperature(probs, self.temperature) if self.temperature != 1.0 else probs