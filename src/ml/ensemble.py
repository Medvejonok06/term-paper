"""Ансамбль, температурне калібрування, RPS і CatBoost із поправкою до ринку."""
import numpy as np
from catboost import CatBoostClassifier, Pool
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from src.features.schema import SHIN_COLS
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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

    from sklearn.linear_model import LogisticRegression


class ShinOnly:
    """Чистий ринок як член ансамблю: стекер сам вирішує, скільки довіряти поправкам моделі."""

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X) -> np.ndarray:
        p = X[SHIN_COLS].to_numpy(dtype=float).copy()
        p[~np.isfinite(p).all(axis=1)] = [0.45, 0.26, 0.29]
        return p / p.sum(axis=1, keepdims=True)


class ColumnLR:
    """Логістична регресія на фіксованому наборі колонок (наприклад, лише elo_diff)."""

    def __init__(self, cols: list[str], C: float = 1.0):
        self.cols = cols
        self.model = LogisticRegression(C=C, max_iter=2000)

    def fit(self, X, y):
        self.model.fit(X[self.cols], y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X[self.cols])

class Sub:
    """Віддає моделі лише вказані колонки, щоб вона не бачила зайвого (наприклад, коефіцієнтів)."""

    def __init__(self, model, cols: list[str]):
        self.model, self.cols = model, cols

    def fit(self, X, y):
        self.model.fit(X[self.cols], y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X[self.cols])


class DistilledMarket:
    """Дистиляція ринку: відтворює ймовірності букмекерів (Shin) лише з фундаментальних ознак.
    Коефіцієнти потрібні тільки при навчанні (як цілі), під час прогнозу їх немає."""

    def __init__(self, cols: list[str], kind: str = "ridge"):
        self.cols, self.kind = cols, kind
        if kind == "ridge":
            self.model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=30.0))
        else:
            self.model = CatBoostRegressor(loss_function="MultiRMSE", depth=3, iterations=400, learning_rate=0.03,
                                           l2_leaf_reg=10.0, random_seed=42, verbose=0, allow_writing_files=False)

    def fit(self, X, y=None):
        p = X[SHIN_COLS].to_numpy(dtype=float)
        ok = np.isfinite(p).all(axis=1)
        p = np.clip(p[ok], 1e-4, 1.0)
        p = p / p.sum(axis=1, keepdims=True)
        z = np.column_stack([np.log(p[:, 0] / p[:, 1]), np.log(p[:, 2] / p[:, 1])])
        self.model.fit(X[self.cols].to_numpy(dtype=float)[ok], z)
        return self

    def predict_proba(self, X) -> np.ndarray:
        z = np.asarray(self.model.predict(X[self.cols].to_numpy(dtype=float)))
        return softmax(np.column_stack([z[:, 0], np.zeros(len(z)), z[:, 1]]))