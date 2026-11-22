"""Elo-рейтинг з урахуванням різниці голів (MOV) та регресією до середнього між сезонами."""
import numpy as np

HOME_ADVANTAGE = 65.0


class MovEloCalculator:
    def __init__(self, k_factor: float = 20.0, home_advantage: float = HOME_ADVANTAGE,
                 initial_elo: float = 1500.0):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_elo = initial_elo
        self.ratings: dict[int, float] = {}

    def get_rating(self, team_id: int) -> float:
        return self.ratings.get(team_id, self.initial_elo)

    def new_season(self, regress: float = 0.25) -> None:
        """Між сезонами команди частково «повертаються» до середнього (склад, трансфери)."""
        for team_id in self.ratings:
            self.ratings[team_id] = (1 - regress) * self.ratings[team_id] + regress * self.initial_elo

    def update_match(self, home_id: int, away_id: int, home_goals: int, away_goals: int) -> None:
        r_home, r_away = self.get_rating(home_id), self.get_rating(away_id)
        expected_home = 1.0 / (1.0 + 10.0 ** ((r_away - (r_home + self.home_advantage)) / 400.0))

        if home_goals > away_goals:
            actual_home = 1.0
        elif home_goals == away_goals:
            actual_home = 0.5
        else:
            actual_home = 0.0

        mov_multiplier = np.sqrt(abs(home_goals - away_goals) + 1.0)
        delta = self.k_factor * mov_multiplier * (actual_home - expected_home)
        self.ratings[home_id] = r_home + delta
        self.ratings[away_id] = r_away - delta