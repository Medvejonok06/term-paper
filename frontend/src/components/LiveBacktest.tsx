import { useEffect, useState } from 'react';
import { api, errorMessage } from '../api';
import type { LiveSummary } from '../types';

export default function LiveBacktest() {
  const [s, setS] = useState<LiveSummary | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api.live().then(setS).catch((e) => setErr(errorMessage(e))); }, []);

  return (
    <div className="max-w-5xl mx-auto mt-8 bg-slate-900/70 border border-slate-800 rounded-2xl p-6">
      <h3 className="text-sm font-bold text-slate-100">Живий бектест (прогнози зафіксовано до матчів)</h3>
      {err && <p className="text-[11px] text-slate-500 mt-2">{err}. Запускайте log_predictions до туру, а evaluate після нього.</p>}
      {s && (
        <>
          <p className="text-[11px] text-slate-400 mt-1 mb-4">
            Зафіксовано {s.logged}, завершено {s.settled}, з коефіцієнтами {s.with_odds}. На малій вибірці інтервали широкі.
          </p>
          <table className="w-full text-xs text-slate-300">
            <thead className="text-slate-400 font-mono text-[10px] uppercase">
              <tr><th className="p-2 text-left">Модель</th><th className="p-2 text-left">n</th><th className="p-2 text-left">Accuracy</th>
                <th className="p-2 text-left">Log loss</th><th className="p-2 text-left">RPS</th></tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {s.models.map((m) => (
                <tr key={m.name}><td className="p-2">{m.name}</td><td className="p-2 font-mono">{m.n}</td>
                  <td className="p-2 font-mono">{m.accuracy}%</td><td className="p-2 font-mono">{m.log_loss}</td>
                  <td className="p-2 font-mono">{m.rps}</td></tr>
              ))}
            </tbody>
          </table>
          <div className="mt-4 space-y-1">
            {s.comparisons.map((c) => {
              const ci = c.lo !== null && c.hi !== null;
              const sig = ci && (c.lo! > 0 || c.hi! < 0);
              return (
                <div key={c.name} className="text-[11px] text-slate-400">
                  {c.name} (n={c.n}): <span className="font-mono text-slate-200">{c.diff > 0 ? '+' : ''}{c.diff}</span>
                  {ci ? ` · 95% ДІ [${c.lo}, ${c.hi}] · ${sig ? 'значуща' : 'не значуща'}` : ' · мало матчів для інтервалу'}
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}