from datetime import date, datetime
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
