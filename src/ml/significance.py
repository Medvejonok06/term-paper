"""Парний бутстреп різниці log loss на тесті. Від'ємне значення = наша модель краща."""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

SHIN = ["shin_prob_home", "shin_prob_draw", "shin_prob_away"]


def main(n_boot: int = 5000) -> None:
    df = pd.read_parquet("data/processed/epl_features.parquet").dropna(subset=["target"])
    tr, te = df[df.split == "train"], df[df.split == "test"]
    y = te["target"].astype(int).to_numpy()
    rng = np.random.default_rng(0)

    def per_match(p):
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1))

    def ours(key):
        b = joblib.load(f"models/bundle_{key}.joblib")
        return per_match(b["ensemble"].predict_proba(te[b["cols"]]))

    shin = te[SHIN].to_numpy()
    book = per_match(shin / shin.sum(1, keepdims=True))
    elo = per_match(LogisticRegression().fit(tr[["elo_diff"]], tr["target"].astype(int)).predict_proba(te[["elo_diff"]]))

    def ci(a, b, name):
        d = a - b
        boots = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        verdict = "значуща" if lo > 0 or hi < 0 else "НЕ значуща"
        print(f"{name}: {d.mean():+.4f}  95% ДІ [{lo:+.4f}, {hi:+.4f}]  {verdict}")

    ci(ours("market"), book, "наш (з коеф.) − букмекери       ")
    ci(ours("fundamental"), elo, "наш (без коеф.) − Elo-регресія  ")
    ci(ours("fundamental"), book, "наш (без коеф.) − букмекери     ")


if __name__ == "__main__":
    main()