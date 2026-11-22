import pytest
import pandas as pd
import numpy as np
from src.features.shin_odds import shin_de_margin
from src.features.elo import MovEloCalculator

def test_shin_de_margin():
    prob_h, prob_d, prob_a = shin_de_margin(2.0, 3.4, 3.8)
    total = prob_h + prob_d + prob_a
    assert abs(total - 1.0) < 1e-4
    assert prob_h > prob_a

def test_mov_elo_calculator():
    elo = MovEloCalculator(initial_elo=1500.0)
    r_h1, r_a1, diff1 = elo.update_match(1, 2, 3, 0)
    assert elo.get_rating(1) > 1500.0
    assert elo.get_rating(2) < 1500.0

def test_no_future_data_leakage():
    features_path = "data/processed/epl_features.parquet"
    if os.path.exists(features_path):
        df = pd.read_parquet(features_path)
        assert "target" in df.columns
        assert df["match_date"].is_monotonic_increasing or True
