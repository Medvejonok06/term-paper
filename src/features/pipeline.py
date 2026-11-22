import os
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from src.db.session import SessionLocal
from src.db.models import Match, MatchStats, Team
from src.features.shin_odds import shin_de_margin
from src.features.elo import MovEloCalculator
from src.features.dixon_coles import predict_score_matrix

def build_features_dataset():
    db = SessionLocal()
    print("--> Querying database for feature engineering...")
    
    query = db.query(Match, MatchStats).join(MatchStats, Match.id == MatchStats.match_id).order_by(Match.match_date, Match.id)
    records = query.all()

    rows = []
    for match, stats in records:
        rows.append({
            "match_id": match.id,
            "match_date": match.match_date,
            "season": match.season,
            "home_team_id": match.home_team_id,
            "away_team_id": match.away_team_id,
            "home_score": match.home_score,
            "away_score": match.away_score,
            "result_1x2": match.result_1x2,
            "home_xg": stats.home_xg or 0.0,
            "away_xg": stats.away_xg or 0.0,
            "home_shots": stats.home_shots or 0,
            "away_shots": stats.away_shots or 0,
            "home_shots_target": stats.home_shots_target or 0,
            "away_shots_target": stats.away_shots_target or 0,
            "b365h": stats.b365_home_win_odds,
            "b365d": stats.b365_draw_odds,
            "b365a": stats.b365_away_win_odds,
        })
    db.close()

    df = pd.DataFrame(rows)
    print(f"Total raw matches loaded: {len(df)}")

    # 1. Shin Odds De-margining
    print("--> Computing Shin implied probabilities...")
    shin_probs = [shin_de_margin(h, d, a) for h, d, a in zip(df['b365h'], df['b365d'], df['b365a'])]
    df['shin_prob_home'] = [p[0] for p in shin_probs]
    df['shin_prob_draw'] = [p[1] for p in shin_probs]
    df['shin_prob_away'] = [p[2] for p in shin_probs]

    # 2. MOV-Elo Ratings
    print("--> Computing MOV-Elo Ratings...")
    elo_calc = MovEloCalculator()
    elo_home_list, elo_away_list, elo_diff_list = [], [], []

    for idx, row in df.iterrows():
        h_id, a_id = row['home_team_id'], row['away_team_id']
        r_h = elo_calc.get_rating(h_id)
        r_a = elo_calc.get_rating(a_id)
        
        elo_home_list.append(r_h)
        elo_away_list.append(r_a)
        elo_diff_list.append(r_h - r_a + elo_calc.home_advantage)
        
        elo_calc.update_match(h_id, a_id, row['home_score'], row['away_score'])

    df['elo_home'] = elo_home_list
    df['elo_away'] = elo_away_list
    df['elo_diff'] = elo_diff_list

    # 3. Rolling EWMA Stats (3, 5, 10 matches)
    print("--> Computing Rolling EWMA xG & Stats...")
    for span in [3, 5, 10]:
        df[f'home_xg_ewma_{span}'] = df.groupby('home_team_id')['home_xg'].transform(lambda x: x.shift(1).ewm(span=span).mean()).fillna(1.2)
        df[f'away_xg_ewma_{span}'] = df.groupby('away_team_id')['away_xg'].transform(lambda x: x.shift(1).ewm(span=span).mean()).fillna(1.0)
        df[f'home_xga_ewma_{span}'] = df.groupby('home_team_id')['away_xg'].transform(lambda x: x.shift(1).ewm(span=span).mean()).fillna(1.0)
        df[f'away_xga_ewma_{span}'] = df.groupby('away_team_id')['home_xg'].transform(lambda x: x.shift(1).ewm(span=span).mean()).fillna(1.2)

    # 4. Dixon-Coles expected probabilities
    print("--> Computing Dixon-Coles probabilities...")
    dc_h, dc_d, dc_a = [], [], []
    for idx, row in df.iterrows():
        lambda_val = max(0.2, (row['home_xg_ewma_5'] + row['away_xga_ewma_5']) / 2.0)
        mu_val = max(0.2, (row['away_xg_ewma_5'] + row['home_xga_ewma_5']) / 2.0)
        ph, pd_val, pa, _ = predict_score_matrix(lambda_val, mu_val)
        dc_h.append(ph)
        dc_d.append(pd_val)
        dc_a.append(pa)

    df['dc_prob_home'] = dc_h
    df['dc_prob_draw'] = dc_d
    df['dc_prob_away'] = dc_a

    # Target Mapping: H -> 0, D -> 1, A -> 2
    target_map = {'H': 0, 'D': 1, 'A': 2}
    df['target'] = df['result_1x2'].map(target_map)

    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/epl_features.parquet"
    df.to_parquet(output_path, index=False)
    print(f"✅ Feature Engineering Complete. Saved {len(df)} rows to {output_path}")

if __name__ == "__main__":
    build_features_dataset()
