import { useEffect, useState } from 'react';
import { api, errorMessage } from '../api';
import TeamBadge from '../components/TeamBadge';
import type { Standing } from '../types';

const dot: Record<string, string> = { W: 'bg-emerald-500', D: 'bg-amber-500', L: 'bg-rose-500' };
const zone = (pos: number) =>
  pos <= 4 ? 'border-l-4 border-emerald-400' : pos >= 18 ? 'border-l-4 border-rose-500' : 'border-l-4 border-transparent';

export default function Standings() {
  const [rows, setRows] = useState<Standing[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api.standings().then(setRows).catch((e) => setErr(errorMessage(e))); }, []);

  if (err) return <div className="text-xs text-amber-300">{err}</div>;
  if (!rows) return <div className="text-xs text-slate-500 animate-pulse">Завантаження таблиці...</div>;
  const head = ['#', 'Команда', 'І', 'В', 'Н', 'П', 'Голи', 'РМ', 'О', 'Форма', 'Elo'];
  return (
    <div className="max-w-5xl mx-auto bg-slate-900/70 border border-slate-800 rounded-2xl overflow-x-auto">
      <table className="w-full text-xs text-slate-200">
        <thead className="text-slate-400 font-mono text-[10px] uppercase">
          <tr>{head.map((h) => <th key={h} className="p-3 text-left">{h}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id} className={`border-t border-slate-800/60 ${zone(r.pos)}`}>
              <td className="p-3 font-mono text-slate-400">{r.pos}</td>
              <td className="p-3"><span className="flex items-center gap-2.5 font-semibold">
                <TeamBadge src={r.crest} name={r.team} size={22} />{r.team}</span></td>
              <td className="p-3 font-mono">{r.P}</td><td className="p-3 font-mono">{r.W}</td>
              <td className="p-3 font-mono">{r.D}</td><td className="p-3 font-mono">{r.L}</td>
              <td className="p-3 font-mono">{r.GF}:{r.GA}</td>
              <td className="p-3 font-mono">{r.GD > 0 ? `+${r.GD}` : r.GD}</td>
              <td className="p-3 font-mono font-bold text-white">{r.Pts}</td>
              <td className="p-3"><span className="flex gap-1">
                {r.form.map((c, i) => <span key={i} className={`w-2.5 h-2.5 rounded-full ${dot[c]}`} />)}</span></td>
              <td className="p-3 font-mono text-slate-400">{r.elo ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="p-3 text-[10px] text-slate-500">
        Зелена смуга: зона Ліги чемпіонів, червона: зона вильоту. Джерело: Fantasy Premier League. Elo: наша модель.
      </p>
    </div>
  );
}