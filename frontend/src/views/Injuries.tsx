import { useEffect, useState } from 'react';
import { api, errorMessage } from '../api';
import TeamBadge from '../components/TeamBadge';
import type { InjuryTeam, PlayerInfo } from '../types';

const icon = (s: string) => (s === 's' ? '🟥' : s === 'i' ? '🩹' : s === 'd' ? '⚠️' : s === 'a' ? '✅' : '⛔');
const Line = ({ p }: { p: PlayerInfo }) => (
  <div className="text-[11px] py-0.5 flex gap-2">
    <span>{icon(p.status)}</span><span className="text-[#e8dff0] font-semibold">{p.name}</span>
    <span className="text-[#9a82a5] truncate" title={p.news}>{p.news}</span>
  </div>
);

export default function Injuries() {
  const [data, setData] = useState<InjuryTeam[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api.injuries().then(setData).catch((e) => setErr(errorMessage(e))); }, []);

  if (err) return <div className="text-xs text-amber-300">{err}</div>;
  if (!data) return <div className="text-xs text-[#9a82a5] animate-pulse">Завантаження звіту...</div>;
  return (
    <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
      {data.map((t) => (
        <div key={t.team} className="bg-[#230029]/70 border border-[#3a1245] rounded-2xl p-4">
          <div className="flex items-center justify-between mb-3">
            <span className="flex items-center gap-2 font-bold text-sm"><TeamBadge src={t.crest} name={t.team} size={26} />{t.team}</span>
            <span className="text-[10px] font-mono text-rose-300">−{(t.missing_strength * 100).toFixed(0)}% сили · Elo {t.elo_penalty}</span>
          </div>
          {t.absent.map((p) => <Line key={p.id} p={p} />)}
          {t.doubtful.map((p) => <Line key={p.id} p={p} />)}
          {t.likely_back.map((p) => <Line key={p.id} p={{ ...p, status: 'a' }} />)}
          {t.ban_risk.map((b) => (
            <div key={b.id} className="text-[11px] py-0.5">🟨 <span className="text-[#e8dff0] font-semibold">{b.name}</span>
              <span className="text-[#9a82a5]"> · {b.yellow} з {b.limit} жовтих</span></div>
          ))}
          {!t.absent.length && !t.doubtful.length && !t.ban_risk.length && <div className="text-[11px] text-emerald-300">Усі в строю</div>}
        </div>
      ))}
    </div>
  );
}