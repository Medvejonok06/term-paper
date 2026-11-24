#!/usr/bin/env bash
# Щотижнева процедура ПЕРЕД туром. Зупиняється на першій помилці.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

python -m src.data.sync_football_data | tee /tmp/sync.out
grep -q "Разом додано" /tmp/sync.out || { echo "❌ sync нічого не зробив: перевірте блок __main__"; exit 1; }

python -m src.features.build_features
python - << 'EOF'
import json
s = json.load(open("data/processed/team_state.json"))
print("останній матч у стані:", max(v["last_match"] for v in s.values()))
EOF

pkill -f "uvicorn src.api.main" 2>/dev/null || true
sleep 1
nohup uvicorn src.api.main:app --port 8000 > uvicorn.log 2>&1 &
for _ in $(seq 1 60); do
  curl -sf http://127.0.0.1:8000/api/v1/health > /dev/null && break
  sleep 1
done
curl -sf http://127.0.0.1:8000/api/v1/health > /dev/null || { echo "❌ Бекенд не стартував, дивіться uvicorn.log"; exit 1; }

python -m src.live.log_predictions

python - << 'EOF'
import json, os
path = "data/live/predictions.jsonl"
rows = [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []
f = lambda p: "/".join(f"{v:.0%}" for v in p)
print(f"\nУ журналі прогнозів: {len(rows)}")
worst = 0.0
for d in rows:
    delta = max(abs(a - b) for a, b in zip(d["probs_adj"], d["probs_base"])) * 100
    worst = max(worst, delta)
    print(f"{d['home']:<16}{d['away']:<16} база {f(d['probs_base']):<12} з травмами {f(d['probs_adj']):<12}"
          f" Elo {d['adjustment']['home_elo']:+.0f}/{d['adjustment']['away_elo']:+.0f}")
print(f"Найбільша зміна через травми: {worst:.1f} п.в.")
EOF