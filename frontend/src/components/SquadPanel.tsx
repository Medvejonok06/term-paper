import { useEffect, useState } from 'react';
import { api, errorMessage } from '../api';
import type { PlayerInfo, Squad } from '../types';
import TeamBadge from './TeamBadge';

const LINES: PlayerInfo['pos'][] = ['FWD', 'MID', 'DEF', 'GK'];
const ACCENT = { emerald: 'text-emerald-400', cyan: 'text-cyan-400' } as const;
const icon = (s: string) => (s === 's' ? '🟥' : s === 'i' ? '🩹' : s === 'u' ? '👋' : s === 'd' ? '⚠️' : s === 'a' ? '✅' : '⛔');

function Dot({ p, yellowRisk }: { p: PlayerInfo; yellowRisk: boolean }) {
  const [broken, setBroken] = useState(false);
  const doubtful = p.availability < 1;
  return (
    <div className="flex flex-col items-center w-16">
      <div className="relative">
        <div className={`w-11 h-11 rounded-full overflow-hidden bg-[#3a1245] border-2 ${doubtful ? 'border-amber-400' : 'border-white/70'}`}>
          {p.photo && !broken ? (
            <img src={p.photo} alt={p.name} onError={() => setBroken(true)} referrerPolicy="no-referrer"
              className="w-full h-full object-cover object-top" />
          ) : (
            <div className="w-full h-full flex items-center justify-center text-[11px] font-bold text-[#d5c8dc]">
              {p.name.slice(0, 2).toUpperCase()}
            </div>
          )}
        </div>
        {yellowRisk && (
          <span title="Одна жовта картка до дискваліфікації"
            className="absolute -top-1 -right-1 w-2.5 h-3.5 bg-yellow-400 rounded-[2px] border border-yellow-600" />
        )}
        {doubtful && (
          <span title={p.news}
            className="absolute -bottom-1 -right-1 w-4 h-4 rounded-full bg-amber-500 text-[10px] font-bold text-[#14001a] flex items-center justify-center">?</span>
        )}
      </div>
      <span className="mt-1 max-w-full truncate text-[10px] font-semibold text-white drop-shadow">{p.name}</span>
    </div>
  );
}

function Row({ p }: { p: PlayerInfo }) {
  return (
    <div className="flex items-start gap-2 text-[11px] py-1">
      <span>{icon(p.status)}</span>
      <div className="min-w-0">
        <span className="font-semibold text-[#e8dff0]">{p.name}</span>
        <span className="text-[#9a82a5]"> · {p.pos}{p.chance !== null ? ` · шанс ${p.chance}%` : ''}</span>
        {p.news && <div className="text-[#9a82a5] truncate" title={p.news}>{p.news}</div>}
      </div>
    </div>
  );
}

export default function SquadPanel({ team, accent }: { team: string; accent: keyof typeof ACCENT }) {
  const [squad, setSquad] = useState<Squad | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setSquad(null);
    setErr(null);
    api.squad(team).then((s) => alive && setSquad(s)).catch((e) => alive && setErr(errorMessage(e)));
    return () => { alive = false; };
  }, [team]);

  const card = 'bg-[#230029]/70 border border-[#3a1245] rounded-2xl p-5';
  if (err) return <div className={`${card} text-xs text-[#9a82a5]`}>Склад {team}: {err}</div>;
  if (!squad) return <div className={`${card} text-xs text-[#9a82a5] animate-pulse`}>Завантаження складу {team}...</div>;

  const risk = new Set(squad.ban_risk.map((b) => b.id));
  return (
    <div className={card}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2 min-w-0">
          <TeamBadge src={squad.crest} name={team} size={28} />
          <div className="min-w-0">
            <div className={`text-sm font-bold truncate ${ACCENT[accent]}`}>{team}</div>
            <div className="text-[10px] text-[#9a82a5]">Імовірний склад, схема ≈ {squad.formation}</div>
          </div>
        </div>
        {squad.elo_penalty < 0 && (
          <div className="text-right text-[10px] text-[#b9a5c3]">
            Відсутня сила: <span className="font-mono text-rose-300">{(squad.missing_strength * 100).toFixed(0)}%</span>
            <div className="font-mono text-[#9a82a5]">Elo {squad.elo_penalty}</div>
          </div>
        )}
      </div>

      <div className="relative rounded-xl overflow-hidden border border-emerald-900/60 bg-gradient-to-b from-emerald-800/60 to-emerald-900/70 pitch-bg"
        style={{ aspectRatio: '3 / 3.7' }}>
        <div className="absolute inset-3 border border-white/25 rounded-sm" />
        <div className="absolute left-3 right-3 top-1/2 border-t border-white/25" />
        <div className="absolute left-1/2 top-1/2 w-16 h-16 -translate-x-1/2 -translate-y-1/2 rounded-full border border-white/25" />
        <div className="absolute left-1/4 right-1/4 top-3 h-12 border border-white/25 border-t-0" />
        <div className="absolute left-1/4 right-1/4 bottom-3 h-12 border border-white/25 border-b-0" />
        <div className="absolute inset-0 flex flex-col justify-between py-5 px-2">
          {LINES.map((pos) => (
            <div key={pos} className="flex justify-around">
              {squad.starting_xi.filter((p) => p.pos === pos).map((p) => <Dot key={p.id} p={p} yellowRisk={risk.has(p.id)} />)}
            </div>
          ))}
        </div>
      </div>

      <div className="mt-4 space-y-3">
        {squad.absent.length > 0 && (
          <div>
            <div className="text-[11px] font-semibold text-rose-300 mb-1">Не зіграють</div>
            {squad.absent.map((p) => <Row key={p.id} p={p} />)}
          </div>
        )}
        {squad.doubtful.length > 0 && (
          <div>
            <div className="text-[11px] font-semibold text-amber-300 mb-1">Під питанням</div>
            {squad.doubtful.map((p) => <Row key={p.id} p={p} />)}
          </div>
        )}

                {squad.likely_back.length > 0 && (
          <div>
            <div className="text-[11px] font-semibold text-emerald-300 mb-1">Лікарі: імовірно зіграють</div>
            {squad.likely_back.map((p) => <Row key={p.id} p={{ ...p, status: 'a' }} />)}
          </div>
        )}

        
        {squad.ban_risk.length > 0 && (
          <div>
            <div className="text-[11px] font-semibold text-yellow-300 mb-1">Ризик дискваліфікації (жовті картки)</div>
            {squad.ban_risk.map((b) => (
              <div key={b.id} className="text-[11px] text-[#b9a5c3] py-0.5">
                🟨 <span className="text-[#e8dff0] font-semibold">{b.name}</span>: {b.yellow} з {b.limit}, вікно ще {b.matches_left} матч.
              </div>
            ))}
          </div>
        )}
        {squad.left.length > 0 && (
          <div>
            <div className="text-[11px] font-semibold text-[#b9a5c3] mb-1">Залишили клуб</div>
            {squad.left.map((p) => <Row key={p.id} p={p} />)}
          </div>
        )}
        <div>
          <div className="text-[11px] font-semibold text-[#b9a5c3] mb-1">Лавка (оцінка)</div>
          <div className="flex flex-wrap gap-1.5">
            {squad.bench.map((p) => (
              <span key={p.id} className="px-2 py-0.5 rounded bg-[#3a1245] text-[10px] text-[#d5c8dc]">{p.name}</span>
            ))}
          </div>
        </div>
      </div>

      <p className="mt-4 text-[10px] text-[#7a5a86] leading-relaxed">
        Оцінка за хвилинами, ціною та статусами гравців ({squad.source}, оновлено {squad.updated}). Офіційний склад
        з’явиться безпосередньо перед матчем. Емблеми й фото належать клубам і АПЛ та використовуються лише в навчальному проєкті.
      </p>
    </div>
  );
}