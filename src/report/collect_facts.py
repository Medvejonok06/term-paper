"""Збирає факти для звіту з артефактів: docs/report_facts.md"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import joblib
import pandas as pd


def sh(*cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return "n/a"


def main() -> None:
    m = json.loads(Path("models/metrics.json").read_text(encoding="utf-8"))
    out = ["# Факти для звіту (згенеровано автоматично)\n", f"Коміт: `{sh('git', 'rev-parse', '--short', 'HEAD')}`\n",
           "## Вибірки і метрики",
           f"train={m['train_samples']}, val={m['val_samples']}, test={m['test_samples']} ({m['test_from']} — {m['test_to']})\n",
           "| Модель | Accuracy | Log loss | RPS |", "|---|---|---|---|"]
    out += [f"| {r['name']} | {r['accuracy']}% | {r['log_loss']} | {r['rps']} |" for r in m["models"]]

    out.append("\n## Абляція наборів ознак (OOF log loss)")
    for key, rows in m.get("ablation", {}).items():
        for r in rows:
            out.append(f"- {key}/{r['set']}: {r['n_features']} ознак, OOF {r.get('oof_log_loss')}, ваги {r.get('weights')}")

    out.append("\n## Ансамблі")
    for key in ("fundamental", "market"):
        b = joblib.load(f"models/bundle_{key}.joblib")
        e = b["ensemble"]
        members = ", ".join(f"{type(x).__name__}={w:.2f}" for x, w in zip(e.models, e.weights))
        out += [f"- **{key}**: набір `{b.get('set', '?')}`, {len(b['cols'])} ознак, T={e.temperature:.3f}, "
                f"OOF log loss {b['val_loss']:.4f}", f"  - члени: {members}", f"  - ознаки: {', '.join(b['cols'])}"]

    df = pd.read_parquet("data/processed/epl_features.parquet")
    dates = pd.to_datetime(df["match_date"])
    out += ["\n## Дані", f"Матчів: {len(df)}, сезонів: {df['season'].nunique()}, "
            f"період: {dates.min():%Y-%m-%d} — {dates.max():%Y-%m-%d}"]
    if "split" in df.columns:
        out.append("Розподіл за вибірками: " + str(df["split"].value_counts().to_dict()))
    res = df.groupby("season")["result_1x2"].value_counts(normalize=True).unstack().round(3)
    out += ["\nЧастки результатів за сезонами:", "```", res.to_string(), "```"]
    calib = Path("data/processed/goal_calib.json")
    if calib.exists():
        out.append(f"Калібрування рівня голів: `{calib.read_text()}`")

    log, live = Path("data/live/predictions.jsonl"), Path("data/live/summary.json")
    n_log = sum(1 for line in log.open(encoding="utf-8") if line.strip()) if log.exists() else 0
    out += ["\n## Живий бектест", f"Прогнозів у журналі: {n_log}"]
    if live.exists():
        out += ["```", json.dumps(json.loads(live.read_text(encoding="utf-8")), ensure_ascii=False, indent=2), "```"]
    else:
        out.append("Завершених матчів ще немає (summary.json відсутній).")

    Path("docs").mkdir(exist_ok=True)
    Path("docs/report_facts.md").write_text("\n".join(out), encoding="utf-8")
    print("✅ docs/report_facts.md")


if __name__ == "__main__":
    main()