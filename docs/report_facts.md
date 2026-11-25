# Факти для звіту (згенеровано автоматично)

Коміт: `f52be35`

## Вибірки і метрики
train=2273, val=394, test=364 (2025-11-09 — 2026-11-08)

| Модель | Accuracy | Log loss | RPS |
|---|---|---|---|
| Частоти результатів (без ознак) | 39.56% | 1.103 | 0.2285 |
| Elo + логістична регресія | 47.25% | 1.0443 | 0.2101 |
| Dixon-Coles (xG-форма) | 46.43% | 1.0581 | 0.2148 |
| Букмекери (Bet365, Shin) | 48.08% | 1.0261 | 0.2043 |
| Наш ансамбль (без коефіцієнтів) | 45.6% | 1.0366 | 0.2087 |
| Наш ансамбль (з коефіцієнтами) | 47.53% | 1.0317 | 0.2055 |

## Абляція наборів ознак (OOF log loss)
- fundamental/v1: 26 ознак, OOF 0.9568, ваги {'elo_lr': 0.0, 'cb': 0.2, 'lgbm': 0.0, 'lr_all': 0.3, 'dist_ridge': 0.5, 'dist_cb': 0.0}
- fundamental/lean: 22 ознак, OOF 0.9566, ваги {'elo_lr': 0.0, 'cb': 0.3, 'lgbm': 0.0, 'lr_all': 0.0, 'dist_ridge': 0.7, 'dist_cb': 0.0}
- fundamental/full: 69 ознак, OOF 0.958, ваги {'elo_lr': 0.2, 'cb': 0.3, 'lgbm': 0.0, 'lr_all': 0.0, 'dist_ridge': 0.5, 'dist_cb': 0.0}
- market/market_lean: 25 ознак, OOF 0.9459, ваги {'market': 0.6, 'cb_offset': 0.4}
- market/market_full: 72 ознак, OOF 0.9463, ваги {'market': 0.5, 'cb_offset': 0.5}

## Ансамблі
- **fundamental**: набір `lean`, 22 ознак, T=1.024, OOF log loss 0.9566
  - члени: ColumnLR=0.00, Sub=0.30, Sub=0.00, Sub=0.00, DistilledMarket=0.70, DistilledMarket=0.00
  - ознаки: elo_home, elo_away, elo_diff, xg_diff_5, pts_diff_5, exp_home_goals, exp_away_goals, dc_prob_home, dc_prob_draw, dc_prob_away, sot_diff_5, fin_diff_10, venue_pts_diff, venue_gd_diff, rest_diff, rank_diff, ppg_diff, h2h_ppg, h2h_gd, home_g14, away_g14, season_progress
- **market**: набір `market_lean`, 25 ознак, T=0.963, OOF log loss 0.9459
  - члени: ShinOnly=0.60, MarketOffsetCatBoost=0.40
  - ознаки: shin_prob_home, shin_prob_draw, shin_prob_away, elo_home, elo_away, elo_diff, xg_diff_5, pts_diff_5, exp_home_goals, exp_away_goals, dc_prob_home, dc_prob_draw, dc_prob_away, sot_diff_5, fin_diff_10, venue_pts_diff, venue_gd_diff, rest_diff, rank_diff, ppg_diff, h2h_ppg, h2h_gd, home_g14, away_g14, season_progress

## Дані
Матчів: 3150, сезонів: 9, період: 2018-08-10 — 2026-11-23
Розподіл за вибірками: {'train': 2273, 'val': 394, 'test': 364, 'warmup': 109, 'live': 10}

Частки результатів за сезонами:
```
result_1x2      A      D      H
season                         
1819        0.337  0.187  0.476
1920        0.305  0.242  0.453
2021        0.403  0.218  0.379
2122        0.339  0.232  0.429
2223        0.287  0.229  0.484
2324        0.324  0.216  0.461
2425        0.347  0.245  0.408
2526        0.300  0.274  0.426
2627        0.291  0.273  0.436
```
Калібрування рівня голів: `{"xg_mean": 1.7445546600356931, "home_goals": 1.553968253968254, "away_goals": 1.3012698412698414}`

## Живий бектест
Прогнозів у журналі: 10
Завершених матчів ще немає (summary.json відсутній).