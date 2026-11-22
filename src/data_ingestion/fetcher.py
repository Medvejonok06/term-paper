import pandas as pd
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
