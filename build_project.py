import os
import sys

files = {}

files["src/config.py"] = '''from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/football_predictor"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
'''

files["src/db/base.py"] = '''from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass
'''

files["src/db/session.py"] = '''from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.config import settings
from src.db.base import Base

engine = create_engine(settings.DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
'''

files["src/db/models.py"] = '''from datetime import date, datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import ForeignKey, String, Integer, Float, Date, DateTime, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.db.base import Base

JSON_TYPE = JSONB().with_variant(JSON(), "sqlite")

class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    code: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    logo_url: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    home_matches: Mapped[List["Match"]] = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    away_matches: Mapped[List["Match"]] = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")

class Match(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    season: Mapped[str] = mapped_column(String(10), index=True)
    match_date: Mapped[date] = mapped_column(Date, index=True)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    home_score: Mapped[int] = mapped_column(Integer, nullable=False)
    away_score: Mapped[int] = mapped_column(Integer, nullable=False)
    result_1x2: Mapped[str] = mapped_column(String(1), nullable=False)

    home_team: Mapped["Team"] = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team: Mapped["Team"] = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")

    stats: Mapped[Optional["MatchStats"]] = relationship("MatchStats", back_populates="match", uselist=False, cascade="all, delete-orphan")
    prediction: Mapped[Optional["Prediction"]] = relationship("Prediction", back_populates="match", uselist=False, cascade="all, delete-orphan")

class MatchStats(Base):
    __tablename__ = "match_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), unique=True, nullable=False)

    home_shots: Mapped[Optional[int]] = mapped_column(Integer)
    away_shots: Mapped[Optional[int]] = mapped_column(Integer)
    home_shots_target: Mapped[Optional[int]] = mapped_column(Integer)
    away_shots_target: Mapped[Optional[int]] = mapped_column(Integer)

    home_xg: Mapped[Optional[float]] = mapped_column(Float)
    away_xg: Mapped[Optional[float]] = mapped_column(Float)

    b365_home_win_odds: Mapped[Optional[float]] = mapped_column(Float)
    b365_draw_odds: Mapped[Optional[float]] = mapped_column(Float)
    b365_away_win_odds: Mapped[Optional[float]] = mapped_column(Float)

    match: Mapped["Match"] = relationship("Match", back_populates="stats")

class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), unique=True, nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)

    prob_home_win: Mapped[float] = mapped_column(Float, nullable=False)
    prob_draw: Mapped[float] = mapped_column(Float, nullable=False)
    prob_away_win: Mapped[float] = mapped_column(Float, nullable=False)

    shap_values: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON_TYPE)
    exact_score_matrix: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON_TYPE)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    match: Mapped["Match"] = relationship("Match", back_populates="prediction")

class EloRating(Base):
    __tablename__ = "elo_ratings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False, index=True)
    match_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    elo_score: Mapped[float] = mapped_column(Float, nullable=False)

    team: Mapped["Team"] = relationship("Team")
'''

files["src/data_ingestion/fetcher.py"] = '''import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session
from src.db.session import SessionLocal, init_db
from src.db.models import Team, Match, MatchStats

SEASONS = ["1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]
BASE_URL = "https://www.football-data.co.uk/mmz4281/{}/E0.csv"

def get_or_create_team(db: Session, name: str) -> Team:
    clean_name = name.strip()
    team = db.query(Team).filter(Team.name == clean_name).first()
    if not team:
        team = Team(name=clean_name, code=clean_name[:3].upper())
        db.add(team)
        db.flush()
    return team

def parse_date(date_str: str) -> datetime.date:
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(str(date_str).strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unknown date format: {date_str}")

def calculate_proxy_xg(shots: float, shots_target: float) -> float:
    if pd.isna(shots) or pd.isna(shots_target):
        return 0.0
    off_target = max(0.0, shots - shots_target)
    return round(0.32 * shots_target + 0.04 * off_target, 2)

def fetch_and_load_data():
    init_db()
    db = SessionLocal()
    total_matches = 0

    try:
        for season in SEASONS:
            url = BASE_URL.format(season)
            print(f"--> Downloading season {season}...")
            try:
                df = pd.read_csv(url)
            except Exception as e:
                print(f"Skipping {season}: {e}")
                continue

            df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
            
            for _, row in df.iterrows():
                try:
                    match_date = parse_date(row["Date"])
                    home_team = get_or_create_team(db, row["HomeTeam"])
                    away_team = get_or_create_team(db, row["AwayTeam"])

                    existing = db.query(Match).filter(
                        Match.match_date == match_date,
                        Match.home_team_id == home_team.id,
                        Match.away_team_id == away_team.id
                    ).first()

                    if existing:
                        continue

                    home_score = int(row["FTHG"])
                    away_score = int(row["FTAG"])
                    result = row["FTR"]

                    match = Match(
                        season=season,
                        match_date=match_date,
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                        home_score=home_score,
                        away_score=away_score,
                        result_1x2=result
                    )
                    db.add(match)
                    db.flush()

                    hs = float(row["HS"]) if "HS" in row and pd.notna(row["HS"]) else 0.0
                    aws = float(row["AS"]) if "AS" in row and pd.notna(row["AS"]) else 0.0
                    hst = float(row["HST"]) if "HST" in row and pd.notna(row["HST"]) else 0.0
                    ast = float(row["AST"]) if "AST" in row and pd.notna(row["AST"]) else 0.0

                    home_xg = calculate_proxy_xg(hs, hst)
                    away_xg = calculate_proxy_xg(aws, ast)

                    stats = MatchStats(
                        match_id=match.id,
                        home_shots=int(hs),
                        away_shots=int(aws),
                        home_shots_target=int(hst),
                        away_shots_target=int(ast),
                        home_xg=home_xg,
                        away_xg=away_xg,
                        b365_home_win_odds=float(row["B365H"]) if "B365H" in row and pd.notna(row["B365H"]) else None,
                        b365_draw_odds=float(row["B365D"]) if "B365D" in row and pd.notna(row["B365D"]) else None,
                        b365_away_win_odds=float(row["B365A"]) if "B365A" in row and pd.notna(row["B365A"]) else None
                    )
                    db.add(stats)
                    total_matches += 1

                except Exception:
                    continue

            db.commit()
            print(f"Season {season} imported.")

        print(f"✅ ETL Complete. Matches added: {total_matches}")

    except Exception as e:
        db.rollback()
        print(f"❌ Error during ETL: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    fetch_and_load_data()
'''

files["src/features/shin_odds.py"] = '''import numpy as np

def shin_de_margin(odds_h: float, odds_d: float, odds_a: float):
    """
    Shin's (1992, 1993) method to convert bookmaker odds into implied probabilities
    removing bookmaker margin.
    """
    if not (odds_h and odds_d and odds_a) or min(odds_h, odds_d, odds_a) <= 1.0:
        return 0.333, 0.333, 0.333

    implied = np.array([1.0 / odds_h, 1.0 / odds_d, 1.0 / odds_a])
    S = np.sum(implied)
    
    if abs(S - 1.0) < 1e-5:
        return float(implied[0]), float(implied[1]), float(implied[2])

    # Iterative solver for z (insider proportion)
    z = 0.0
    for _ in range(30):
        f = np.sum(np.sqrt(z**2 + 4 * (1 - z) * (implied / S))) - (2 + z)
        if abs(f) < 1e-6:
            break
        df = np.sum( (2*z - 4*(implied/S)) / (2 * np.sqrt(z**2 + 4 * (1 - z) * (implied / S))) ) - 1.0
        if df == 0:
            break
        z = z - f / df
        z = max(0.0, min(0.4, z))

    prob = (np.sqrt(z**2 + 4 * (1 - z) * (implied / S)) - z) / (2 * (1 - z))
    prob = prob / np.sum(prob)
    return float(prob[0]), float(prob[1]), float(prob[2])
'''

files["src/features/elo.py"] = '''import numpy as np

class MovEloCalculator:
    def __init__(self, k_factor: float = 20.0, home_advantage: float = 65.0, initial_elo: float = 1500.0):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_elo = initial_elo
        self.ratings = {}

    def get_rating(self, team_id: int) -> float:
        return self.ratings.get(team_id, self.initial_elo)

    def update_match(self, home_id: int, away_id: int, home_goals: int, away_goals: int):
        r_home = self.get_rating(home_id)
        r_away = self.get_rating(away_id)

        exp_home = 1.0 / (1.0 + 10.0 ** ((r_away - (r_home + self.home_advantage)) / 400.0))
        
        if home_goals > away_goals:
            s_home = 1.0
        elif home_goals == away_goals:
            s_home = 0.5
        else:
            s_home = 0.0

        goal_diff = abs(home_goals - away_goals)
        mov_mult = np.sqrt(goal_diff + 1.0)

        delta = self.k_factor * mov_mult * (s_home - exp_home)
        self.ratings[home_id] = r_home + delta
        self.ratings[away_id] = r_away - delta

        return r_home, r_away, r_home - r_away + self.home_advantage
'''

files["src/features/dixon_coles.py"] = '''import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

def tau(x: int, y: int, lambda_val: float, mu_val: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1.0 - lambda_val * mu_val * rho
    elif x == 0 and y == 1:
        return 1.0 + lambda_val * rho
    elif x == 1 and y == 0:
        return 1.0 + mu_val * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    else:
        return 1.0

def predict_score_matrix(lambda_val: float, mu_val: float, rho: float = -0.05, max_goals: int = 5):
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            p = poisson.pmf(x, lambda_val) * poisson.pmf(y, mu_val) * tau(x, y, lambda_val, mu_val, rho)
            matrix[x, y] = max(0.0, p)
    
    total_p = np.sum(matrix)
    if total_p > 0:
        matrix /= total_p

    prob_home = np.sum(np.tril(matrix, -1))
    prob_draw = np.sum(np.diag(matrix))
    prob_away = np.sum(np.triu(matrix, 1))

    return prob_home, prob_draw, prob_away, matrix
'''

files["src/features/pipeline.py"] = '''import os
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
'''

# Write all files
for filepath, content in files.items():
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

print("📁 All project files created cleanly.")
