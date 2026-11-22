import { useEffect, useState } from 'react';
import { api, errorMessage } from '../api';
import TeamBadge from '../components/TeamBadge';
import type { SimResult } from '../types';

const Cell = ({ v, color }: { v: number; color: string }) => (
  <div className="flex items-center gap-2 min-w-[110px]">
    <div className="flex-1 h-1.5 bg-slate-950 rounded-full overflow-hidden"><div className={`h-full ${color}`} style={{ width: `${v * 100}%` }} /></div>
    <span className="font-mono w-10 text-right">{(v * 100).toFixed(1)}%</span>
  </div>
);

export default function Simulation() {
  const [data, setData] = useState<SimResult | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api.simulate(3000).then(setData).catch((e) => setErr(errorMessage(e))); }, []);

  if (err) return <div className="text-xs text-amber-300">{err}</div>;
  if (!data) return <div className="text-xs text-slate-500 animate-pulse">Симуляція решти сезону...</div>;
  return (
    <div className="max-w-5xl mx-auto bg-slate-900/70 border border-slate-800 rounded-2xl overflow-x-auto">
      <table className="w-full text-xs text-slate-200">
        <thead className="text-slate-400 font-mono text-[10px] uppercase">
          <tr>{['Команда', 'Очки', 'Сер. очки', 'Чемпіон', 'Топ-4', 'Вильот'].map((h) => <th key={h} className="p-3 text-left">{h}</th>)}</tr>
        </thead>
        <tbody>
          {data.teams.map((t) => (
            <tr key={t.team} className="border-t border-slate-800/60">
              <td className="p-3"><span className="flex items-center gap-2.5 font-semibold">
                <TeamBadge src={t.crest} name={t.team} size={22} />{t.team}{t.assumed && <span title="Немає даних у БД, припущення">*</span>}</span></td>
              <td className="p-3 font-mono">{t.points_now}</td>
              <td className="p-3 font-mono">{t.exp_points}</td>
              <td className="p-3"><Cell v={t.title} color="bg-emerald-500" /></td>
              <td className="p-3"><Cell v={t.top4} color="bg-cyan-500" /></td>
              <td className="p-3"><Cell v={t.relegation} color="bg-rose-500" /></td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="p-3 text-[10px] text-slate-500 leading-relaxed">
        {data.simulations} симуляцій {data.remaining_matches} матчів, що залишились. Голи моделюються за Пуассоном із xG-форми команд, травми не враховано,
        тож це орієнтир, а не прогноз. * команда без історії в БД, використано обережні припущення.
      </p>
    </div>
  );
}