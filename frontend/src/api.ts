import axios from 'axios';
import type {
  Calibration, Health, InjuryTeam, Match, MetricsResponse, NextRound, Prediction, SimResult, Squad, Standing, TeamOption,
} from './types';

const http = axios.create({ baseURL: '/api/v1', timeout: 20000 });

export function errorMessage(e: unknown): string {
  if (axios.isAxiosError(e)) {
    const detail = e.response?.data?.detail;
    if (typeof detail === 'string') return detail;
    if (e.code === 'ERR_NETWORK') return 'Бекенд недоступний. Перевір, що uvicorn запущено на порту 8000.';
    return e.message;
  }
  return 'Невідома помилка';
}

export const api = {
  health: () => http.get<Health>('/health').then((r) => r.data),
  teams: () => http.get<TeamOption[]>('/teams').then((r) => r.data),
  recentMatches: (limit = 40) => http.get<Match[]>('/matches/recent', { params: { limit } }).then((r) => r.data),
  metrics: () => http.get<MetricsResponse>('/analytics/metrics').then((r) => r.data),
  calibration: (model = 'market') => http.get<Calibration>('/analytics/calibration', { params: { model } }).then((r) => r.data),
  squad: (team: string) => http.get<Squad>(`/squad/${encodeURIComponent(team)}`).then((r) => r.data),
  round: (event: number | null) =>
    http.get<NextRound>(event === null ? '/fixtures/next' : `/fixtures/round/${event}`).then((r) => r.data),
  standings: () => http.get<Standing[]>('/standings').then((r) => r.data),
  injuries: () => http.get<InjuryTeam[]>('/injuries').then((r) => r.data),
  simulate: (n = 3000) => http.get<SimResult>('/simulate/season', { params: { n }, timeout: 60000 }).then((r) => r.data),
  predictCustom: (home_team: string, away_team: string) =>
    http.post<Prediction>('/predict/custom', { home_team, away_team }).then((r) => r.data),
  predictMatch: (id: number) => http.post<Prediction>(`/predict/${id}`).then((r) => r.data),
};