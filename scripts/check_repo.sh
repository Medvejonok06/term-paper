#!/usr/bin/env bash
cd "$(dirname "$0")/.."
source .venv/bin/activate
{
echo "== git =="; git status --short | head -20; git log --oneline | head -5; git remote -v
echo; echo "== відстежувані файли: кількість і розподіл по папках =="
git ls-files | wc -l
git ls-files | sed 's|/[^/]*$||' | sort | uniq -c | sort -rn | head -15
echo; echo "== найбільші відстежувані файли (КБ) =="
git ls-files -z | xargs -0 du -k | sort -n | tail -8
echo; echo "== підозрілі файли =="
git ls-files | grep -Ei '\.env|\.db$|\.sqlite|\.joblib|\.parquet|\.pem|secret|token' || echo "немає"
echo; echo "== підозрілий вміст =="
git grep -nIE 'ghp_|github_pat_|AKIA[0-9A-Z]{12}|api[_-]?key *=|password *=' -- . ':!package-lock.json' | head || true
echo; echo "== python compile =="; python -m compileall -q src && echo OK
echo; echo "== frontend tsc =="; (cd frontend && npx tsc --noEmit && echo OK)
echo; echo "== API smoke =="
for p in health teams standings injuries elo analytics/metrics analytics/live; do
  printf "%-20s" "$p"; curl -s -o /dev/null -w "%{http_code}\n" "http://127.0.0.1:8000/api/v1/$p"
done
} 2>&1 | tee check_report.txt
