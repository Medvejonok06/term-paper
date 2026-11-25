# EPL Match Predictor (курсова робота)

Система ймовірнісних прогнозів матчів англійської Прем'єр-ліги: Elo + xG-форма, ансамбль зі стекінгом,
калібрування, пояснення (SHAP), склади й травми (FPL API), веб-інтерфейс і журнал живих прогнозів.

## Структура

    src/db/            моделі БД (SQLAlchemy)
    src/data/          sync з football-data.co.uk, клієнт FPL API
    src/features/      Elo, Shin, Діксон-Коулз, побудова ознак
    src/ml/            навчання (walk-forward стекінг), ансамбль, тест значущості
    src/api/           FastAPI (17 ендпоінтів, документація на /docs)
    src/live/          журнал живих прогнозів і живий бектест
    frontend/          React + Vite + Tailwind
    scripts/           щотижнева процедура, перевірка репозиторію

## Запуск

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python -m src.features.build_features
    python -m src.ml.train
    uvicorn src.api.main:app --port 8000
    cd frontend && npm install && npm run dev      # http://localhost:3000

## Чесні обмеження

- На тесті (364 матчі) модель статистично не відрізняється від букмекерів і від Elo-регресії.
- Евристика травм (поправка до Elo за даними FPL) не перевірялась на історичних даних.
- xG для сезону 2026/27 оцінений регресією на ударах, для старих сезонів реальний.
- FPL API неофіційний. Емблеми й фото належать клубам та АПЛ.
- Прогнози не є порадою для ставок.

Журнал живих прогнозів: `data/live/`. Контрольні суми замороженої моделі: `models/CHECKSUMS.txt`.

## База даних (PostgreSQL)

    docker compose up -d db                          # або власний Postgres; адреса в .env (див. .env.example)
    python -m scripts.import_db                      # таблиці + дані з data/db_export/*.csv
    python -m scripts.export_db                      # зворотній експорт (оновити CSV)

Оновлення свіжими матчами: `python -m src.data.sync_football_data`.
