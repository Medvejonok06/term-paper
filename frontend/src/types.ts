export type Outcome = 'H' | 'D' | 'A';

export interface TeamOption { name: string; short: string; crest: string | null }

export interface TeamContext {
  name: string; elo: number; xgf_5: number; xga_5: number; pts_5: number; crest: string | null;
}

export interface Prediction {
  match_id: number | null;
  home_team: string;
  away_team: string;
  prob_home_win: number;
  prob_draw: number;
  prob_away_win: number;
  predicted_class: Outcome;
  predicted_outcome: string;
  model_used: 'market' | 'fundamental';
  expected_home_goals: number;
  expected_away_goals: number;
  exact_scores: { score: string; prob: number }[];
  score_matrix: number[][];
  over_25: number;
  btts: number;
  shap_factors: { feature: string; value: number | null; impact: number }[];
  home_context: TeamContext;
  away_context: TeamContext;
  home_form: string[];
  away_form: string[];
  h2h: number[][] | null;
  bookmaker_probs: number[] | null;
  actual: { score: string; result: Outcome; correct: boolean } | null;
  split: string | null;
  adjustment: { home_elo: number; away_elo: number } | null;
  data_until: string | null;
  base_probs: number[] | null;
  likely_score: string;
}

export interface Match {
  id: number; season: string; match_date: string; home_team: string; away_team: string;
  home_score: number; away_score: number; result_1x2: Outcome; split: string;
}

export interface ModelMetric {
  key: string; name: string; kind: 'baseline' | 'benchmark' | 'ours';
  accuracy: number; log_loss: number; rps: number;
}

export interface MetricsResponse {
  train_samples: number; val_samples: number; test_samples: number;
  test_from: string; test_to: string; models: ModelMetric[];
}

export interface PlayerInfo {
  id: number; name: string; pos: 'GK' | 'DEF' | 'MID' | 'FWD'; photo: string | null;
  status: string; chance: number | null; news: string; yellow: number; red: number;
  minutes: number; role: 'core' | 'rotation' | 'squad'; availability: number;
}

export interface BanRisk { id: number; name: string; yellow: number; limit: number; matches_left: number }

export interface Squad {
  team: string; crest: string | null; formation: string;
  starting_xi: PlayerInfo[]; bench: PlayerInfo[]; absent: PlayerInfo[]; doubtful: PlayerInfo[]; left: PlayerInfo[]; likely_back: PlayerInfo[];
  ban_risk: BanRisk[]; missing_strength: number; elo_penalty: number; games_played: number;
  source: string; updated: string;
}

export interface Fixture {
  kickoff: string | null; home_team: string; away_team: string;
  home_crest: string | null; away_crest: string | null;
  prob_home_win: number | null; prob_draw: number | null; prob_away_win: number | null;
  predicted_class: Outcome | null; note: string | null;
}

export interface NextRound { event: number; fixtures: Fixture[] }

export interface Standing {
  id: number; pos: number; team: string; name: string; crest: string | null;
  P: number; W: number; D: number; L: number; GF: number; GA: number; GD: number; Pts: number;
  form: string[]; elo: number | null;
}
export interface InjuryTeam {
  team: string; crest: string | null; missing_strength: number; elo_penalty: number;
  absent: PlayerInfo[]; doubtful: PlayerInfo[]; ban_risk: BanRisk[]; likely_back: PlayerInfo[];
}
export interface SimTeam {
  team: string; crest: string | null; points_now: number; exp_points: number; exp_position: number;
  title: number; top4: number; relegation: number; assumed: boolean;
}
export interface SimResult { simulations: number; remaining_matches: number; teams: SimTeam[] }
export interface CalibPoint { pred: number; obs: number; n: number }
export interface Calibration { n: number; ece: number; classes: Record<'H' | 'D' | 'A', CalibPoint[]> }
export interface Health { status: string; matches: number; teams: number; fpl_available: boolean; missing_teams: string[] }
export interface LiveSummary {
  logged: number; settled: number; with_odds: number;
  models: { name: string; n: number; accuracy: number; log_loss: number; rps: number }[];
  comparisons: { name: string; n: number; diff: number; lo: number | null; hi: number | null }[];
}