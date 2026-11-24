import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import {
  Activity, ArrowLeftRight, Award, CalendarDays, Check, CheckCircle2, ChevronDown, ChevronLeft, ChevronRight,
  HeartPulse, History, Info, Search, Shuffle, Sliders, TrendingUp, Trophy, XCircle, Zap,
} from 'lucide-react';
import { api, errorMessage } from './api';
import CalibrationChart from './components/CalibrationChart';
import ScoreHeatmap from './components/ScoreHeatmap';
import SquadPanel from './components/SquadPanel';
import TeamBadge from './components/TeamBadge';
import Injuries from './views/Injuries';
import Simulation from './views/Simulation';
import Standings from './views/Standings';
import type { Fixture, Match, MetricsResponse, NextRound, Outcome, Prediction, TeamOption } from './types';
import LiveBacktest from './components/LiveBacktest';

/* ---------- Допоміжне ---------- */
const pct = (x: number, d = 1) => `${(x * 100).toFixed(d)}%`;
const fmtDate = (iso: string) =>
  new Date(iso).toLocaleDateString('uk-UA', { day: '2-digit', month: 'short', year: 'numeric' });

const OUTCOME: Record<Outcome, { short: string; text: string; box: string; bar: string; ring: string; badge: string }> = {
  H: { short: 'П1', text: 'text-emerald-400', box: 'bg-emerald-950/30 border-emerald-800/60', bar: 'bg-emerald-500', ring: 'ring-2 ring-emerald-500/70', badge: 'bg-emerald-900/60 text-emerald-300' },
  D: { short: 'X', text: 'text-amber-400', box: 'bg-amber-950/30 border-amber-800/60', bar: 'bg-amber-500', ring: 'ring-2 ring-amber-500/70', badge: 'bg-amber-900/60 text-amber-300' },
  A: { short: 'П2', text: 'text-cyan-400', box: 'bg-cyan-950/30 border-cyan-800/60', bar: 'bg-cyan-500', ring: 'ring-2 ring-cyan-500/70', badge: 'bg-cyan-900/60 text-cyan-300' },
};

const FEATURE_LABELS: Record<string, string> = {
  elo_home: 'Elo господарів', elo_away: 'Elo гостей', elo_diff: 'Різниця Elo (з домашньою перевагою)',
  xg_diff_5: 'Різниця xG-балансу (5 м)', pts_diff_5: 'Різниця форми (очки, 5 м)',
  sot_diff_5: 'Різниця ударів у площину (5 м)', fin_diff_10: 'Різниця реалізації (10 м)',
  venue_pts_diff: 'Форма: господарі вдома vs гості на виїзді',
  venue_gd_diff: 'Різниця голів: вдома vs на виїзді',
  rest_diff: 'Різниця днів відпочинку', rank_diff: 'Перевага в турнірній таблиці',
  ppg_diff: 'Різниця очок за матч (сезон)', season_progress: 'Етап сезону',
  h2h_n: 'Кількість очних зустрічей', h2h_ppg: 'H2H: очки господарів за матч', h2h_gd: 'H2H: різниця голів',
  exp_home_goals: 'Очікувані голи господарів', exp_away_goals: 'Очікувані голи гостей',
  dc_prob_home: 'Dixon-Coles: П1', dc_prob_draw: 'Dixon-Coles: нічия', dc_prob_away: 'Dixon-Coles: П2',
  shin_prob_home: 'Ринок (Shin): П1', shin_prob_draw: 'Ринок (Shin): нічия', shin_prob_away: 'Ринок (Shin): П2',
};
const STAT_LABELS: Record<string, string> = {
  xgf: 'xG створено', xga: 'xG пропущено', pts: 'очки', gf: 'голи забито', ga: 'голи пропущено',
  sotf: 'удари в площину', sota: 'удари в площину (проп.)', cs: 'сухі матчі',
  fts: 'матчі без гола', fin: 'реалізація (голи − xG)',
};
const SIMPLE_LABELS: Record<string, string> = {
  v_pts: 'очки у своїй ролі', v_gf: 'голи у своїй ролі', v_ga: 'пропущені у своїй ролі',
  rest: 'днів відпочинку', g14: 'матчів за 14 днів', g30: 'матчів за 30 днів',
  ppg: 'очки за матч (сезон)', rank: 'місце в таблиці', gdpg: 'різниця голів за матч',
};

function featureLabel(name: string): string {
  if (FEATURE_LABELS[name]) return FEATURE_LABELS[name];
  const side = (s: string) => (s === 'home' ? 'Госп.' : 'Гості');
  const m = name.match(/^(home|away)_(xgf|xga|pts|gf|ga|sotf|sota|cs|fts|fin)_(\d+)$/);
  if (m) return `${side(m[1])}: ${STAT_LABELS[m[2]]} (${m[3]} м)`;
  const v = name.match(/^(home|away)_(v_pts|v_gf|v_ga|rest|g14|g30|ppg|rank|gdpg)$/);
  if (v) return `${side(v[1])}: ${SIMPLE_LABELS[v[2]]}`;
  return name;
}

const card = 'bg-[#230029]/70 border border-[#3a1245] rounded-2xl';

/* ---------- Компонент сучасного кастомного Дропдауну ---------- */
function CustomSelect({ value, onChange, options, disabledOption }: {
  value: string; onChange: (val: string) => void; options: TeamOption[]; disabledOption?: string;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onDown = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setIsOpen(false); };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setIsOpen(false); };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onDown); document.removeEventListener('keydown', onKey); };
  }, []);

  const current = options.find((o) => o.name === value);
  return (
    <div className="relative" ref={ref}>
      <button type="button" onClick={() => setIsOpen((v) => !v)}
        className="w-full bg-[#14001a]/80 border border-[#52275f]/80 hover:border-[#ff2e7e]/80 focus:border-[#ff2e7e] focus:ring-1 focus:ring-[#ff2e7e]/50 rounded-xl px-3.5 py-2.5 text-xs text-white font-medium flex items-center justify-between transition-all shadow-sm group">
        <span className="flex items-center gap-2.5 min-w-0">
          {current && <TeamBadge src={current.crest} name={current.name} size={22} />}
          <span className={value ? 'text-[#f5eff9] font-semibold truncate' : 'text-[#9a82a5]'}>{value || 'Оберіть команду'}</span>
        </span>
        <ChevronDown className={`w-4 h-4 text-[#b9a5c3] group-hover:text-[#e8dff0] transition-transform duration-200 ${isOpen ? 'rotate-180 text-[#ff6ba3]' : ''}`} />
      </button>

      {isOpen && (
        <div className="absolute left-0 right-0 mt-1.5 z-50 bg-[#230029]/95 backdrop-blur-xl border border-[#52275f]/80 rounded-xl shadow-2xl shadow-black/90 max-h-64 overflow-y-auto p-1.5 space-y-0.5">
          {options.map((t) => {
            const selected = t.name === value;
            const disabled = t.name === disabledOption;
            return (
              <button key={t.name} type="button" disabled={disabled}
                onClick={() => { onChange(t.name); setIsOpen(false); }}
                className={`w-full text-left px-3 py-2 rounded-lg text-xs font-medium transition-all flex items-center justify-between ${
                  selected ? 'bg-[#e90052] text-white font-semibold'
                  : disabled ? 'opacity-30 cursor-not-allowed text-[#9a82a5]'
                  : 'text-[#d5c8dc] hover:bg-[#3a1245]/80 hover:text-white'}`}>
                <span className="flex items-center gap-2.5"><TeamBadge src={t.crest} name={t.name} size={20} />{t.name}</span>
                {selected && <Check className="w-3.5 h-3.5 text-white" />}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ---------- Решта компонентів ---------- */
function Spinner({ text }: { text: string }) {
  return (
    <div className={`${card} p-20 flex flex-col items-center gap-3 text-xs text-[#b9a5c3] font-mono`}>
      <div className="w-8 h-8 border-2 border-[#ff2e7e] border-t-transparent rounded-full animate-spin" />
      <span>{text}</span>
    </div>
  );
}

function ProbabilityPanel({ p }: { p: Prediction }) {
  const items: { key: Outcome; title: string; value: number; book?: number }[] = [
    { key: 'H', title: `П1 · ${p.home_team}`, value: p.prob_home_win, book: p.bookmaker_probs?.[0] },
    { key: 'D', title: 'Нічия', value: p.prob_draw, book: p.bookmaker_probs?.[1] },
    { key: 'A', title: `П2 · ${p.away_team}`, value: p.prob_away_win, book: p.bookmaker_probs?.[2] },
  ];
  return (
    <div>
      <div className="h-3.5 w-full bg-[#14001a] rounded-full overflow-hidden flex gap-0.5 p-0.5 border border-[#3a1245] mb-5">
        {items.map((i) => (
          <div key={i.key} style={{ width: `${i.value * 100}%` }} className={`${OUTCOME[i.key].bar} h-full rounded-full`} />
        ))}
      </div>
      <div className="grid grid-cols-3 gap-3">
        {items.map((i) => {
          const s = OUTCOME[i.key];
          return (
            <div key={i.key} className={`rounded-xl border p-4 text-center ${s.box} ${p.predicted_class === i.key ? s.ring : ''}`}>
              <div className={`text-xs font-semibold mb-1 truncate ${s.text}`}>{i.title}</div>
              <div className={`text-3xl font-extrabold ${s.text}`}>{pct(i.value)}</div>
              {i.book !== undefined && (
                <div className="mt-1 text-[10px] text-[#9a82a5]">Букмекер: {pct(i.book)}</div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function CompareRow({ label, home, away, floor = 0, digits = 2 }: {
  label: string; home: number; away: number; floor?: number; digits?: number;
}) {
  const h = Math.max(home - floor, 0.0001);
  const a = Math.max(away - floor, 0.0001);
  const share = (h / (h + a)) * 100;
  return (
    <div>
      <div className="flex justify-between text-[11px] font-mono text-[#d5c8dc] mb-1">
        <span>{home.toFixed(digits)}</span>
        <span className="text-[#9a82a5] font-sans">{label}</span>
        <span>{away.toFixed(digits)}</span>
      </div>
      <div className="h-1.5 bg-cyan-600/70 rounded-full overflow-hidden">
        <div className="h-full bg-emerald-500" style={{ width: `${share}%` }} />
      </div>
    </div>
  );
}

function FormStrip({ form }: { form: string[] }) {
  const style: Record<string, string> = { W: 'bg-emerald-500', D: 'bg-amber-500', L: 'bg-rose-500' };
  const letter: Record<string, string> = { W: 'В', D: 'Н', L: 'П' };
  if (form.length === 0) return <span className="text-[10px] text-[#7a5a86]">немає даних</span>;
  return (
    <div className="flex gap-1">
      {form.map((c, i) => (
        <span key={i} className={`w-5 h-5 rounded text-[10px] font-bold text-[#14001a] flex items-center justify-center ${style[c] ?? 'bg-[#7a5a86]'}`}>
          {letter[c] ?? '?'}
        </span>
      ))}
    </div>
  );
}

function TeamCompare({ p }: { p: Prediction }) {
  const h = p.home_context;
  const a = p.away_context;
  return (
    <div className={`${card} p-5`}>
      <h3 className="text-sm font-bold text-[#f5eff9] mb-3 flex items-center gap-2">
        <Activity className="w-4 h-4 text-emerald-400" /> Порівняння команд
      </h3>
      <div className="flex justify-between items-start gap-2 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <TeamBadge src={h.crest} name={h.name} size={26} />
            <span className="text-[11px] font-semibold text-emerald-400">{h.name}</span>
          </div>
          <div className="mt-1.5"><FormStrip form={p.home_form} /></div>
        </div>
        <div className="flex flex-col items-end">
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-semibold text-cyan-400">{a.name}</span>
            <TeamBadge src={a.crest} name={a.name} size={26} />
          </div>
          <div className="mt-1.5"><FormStrip form={p.away_form} /></div>
        </div>
      </div>
      <div className="space-y-3.5">
        <CompareRow label="Рейтинг Elo" home={h.elo} away={a.elo} floor={1300} digits={0} />
        <CompareRow label="xG створено (EWMA 5)" home={h.xgf_5} away={a.xgf_5} />
        <CompareRow label="xG пропущено (EWMA 5)" home={h.xga_5} away={a.xga_5} />
        <CompareRow label="Очки за матч (форма)" home={h.pts_5} away={a.pts_5} />
      </div>
      <div className="mt-4 pt-3 border-t border-[#3a1245] grid grid-cols-2 text-center text-[11px] text-[#b9a5c3]">
        <div>Очікувані голи: <span className="font-mono text-[#e8dff0]">{p.expected_home_goals.toFixed(2)}</span></div>
        <div>Очікувані голи: <span className="font-mono text-[#e8dff0]">{p.expected_away_goals.toFixed(2)}</span></div>
      </div>
      {p.h2h && p.h2h.length > 0 && (
        <div className="mt-4 pt-3 border-t border-[#3a1245]">
          <div className="text-[11px] text-[#9a82a5] mb-1.5">Останні очні зустрічі (рахунок з погляду {h.name})</div>
          <div className="flex gap-1.5 flex-wrap">
            {p.h2h.map(([x, y], i) => (
              <span key={i} className="font-mono text-[11px] px-2 py-0.5 rounded bg-[#3a1245] text-[#e8dff0]">{x}:{y}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ScoreList({ p }: { p: Prediction }) {
  const max = p.exact_scores[0]?.prob || 1;
  return (
    <div className={`${card} p-5`}>
      <h3 className="text-sm font-bold text-slate-100 mb-1 flex items-center gap-2">
        <Award className="w-4 h-4 text-amber-400" /> Найімовірніші рахунки
      </h3>
      <p className="text-[11px] text-slate-400 mb-4 leading-relaxed">
        Для прогнозованого результату ({OUTCOME[p.predicted_class].short}) найімовірніший рахунок:{' '}
        <span className="font-mono font-bold text-white">{p.likely_score}</span>. Окремий рахунок майже завжди
        малоймовірний, а ймовірність перемоги — це сума багатьох рахунків.
      </p>
      <div className="space-y-2.5">
        {p.exact_scores.map((s, idx) => (
          <div key={s.score}>
            <div className="flex justify-between text-xs mb-1">
              <span className={idx === 0 ? 'font-mono font-bold text-white' : 'font-mono text-slate-300'}>{s.score}</span>
              <span className="font-mono text-slate-400">{pct(s.prob)}</span>
            </div>
            <div className="h-1.5 bg-slate-950 rounded-full overflow-hidden">
              <div className="h-full bg-indigo-500 rounded-full" style={{ width: `${(s.prob / max) * 100}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ShapChart({ p }: { p: Prediction }) {
  const data = useMemo(
    () => p.shap_factors.map((f) => ({ name: featureLabel(f.feature), impact: f.impact, value: f.value })),
    [p],
  );
  return (
    <div className={`${card} p-5`}>
      <h3 className="text-sm font-bold text-[#f5eff9] flex items-center gap-2">
        <Info className="w-4 h-4 text-[#ff6ba3]" /> Чому такий прогноз (SHAP)
      </h3>
      <p className="text-[11px] text-[#b9a5c3] mt-1 mb-3 leading-relaxed">
        <span className="text-emerald-400">Зелені</span> ознаки підвищують шанс результату «{p.predicted_outcome}»,{' '}
        <span className="text-rose-400">червоні</span> знижують. Пояснюється CatBoost-складова ансамблю.
      </p>
      {data.length === 0 ? (
        <div className="text-xs text-[#9a82a5] py-8 text-center">Пояснення недоступне</div>
      ) : (
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout="vertical" margin={{ left: 0, right: 12 }}>
              <XAxis type="number" hide />
              <YAxis dataKey="name" type="category" width={175} tick={{ fill: '#b9a5c3', fontSize: 10 }} />
              <Tooltip
                cursor={{ fill: 'rgba(148,163,184,0.08)' }}
                contentStyle={{ backgroundColor: '#230029', borderColor: '#3a1245', borderRadius: 8, fontSize: 11 }}
                formatter={(v: unknown) => [(Number(v) > 0 ? '+' : '') + Number(v).toFixed(3), 'Вплив']}
              />
              <Bar dataKey="impact" radius={[0, 4, 4, 0]}>
                {data.map((d, i) => <Cell key={i} fill={d.impact >= 0 ? '#10b981' : '#f43f5e'} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

function PredictionView({ p }: { p: Prediction }) {
  const s = OUTCOME[p.predicted_class];
  const confidence = Math.max(p.prob_home_win, p.prob_draw, p.prob_away_win);
  const live = p.match_id === null;
  const sign = (x: number) => (x > 0 ? `+${x}` : `${x}`);
  return (
    <div className="space-y-6">
      <div className={`${card} hero p-6`}>
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-5 mb-5 border-b border-[#3a1245]">
          <div>
            <div className="text-[11px] text-[#9a82a5] font-mono mb-2">
              {p.model_used === 'market' ? 'Модель з коефіцієнтами ринку' : 'Модель на основі форми, Elo та xG'}
            </div>
            <div className="flex items-center gap-3 flex-wrap">
              <TeamBadge src={p.home_context.crest} name={p.home_team} size={48} />
              <h2 className="text-3xl font-extrabold text-white">
                {p.home_team} <span className="text-[#9a82a5] text-lg font-normal">vs</span> {p.away_team}
              </h2>
              <TeamBadge src={p.away_context.crest} name={p.away_team} size={48} />
            </div>
          </div>
          <div className={`rounded-xl px-4 py-2.5 border ${s.box}`}>
            <div className="text-[10px] uppercase tracking-wider text-[#b9a5c3]">Прогноз моделі</div>
            <div className={`text-sm font-bold flex items-center gap-1.5 ${s.text}`}>
              <CheckCircle2 className="w-4 h-4" /> {p.predicted_outcome}
            </div>
            <div className="text-[10px] text-[#9a82a5] mt-0.5">впевненість {pct(confidence, 0)}</div>
          </div>
        </div>

        {p.actual && (
          <div className={`mb-5 flex items-center gap-2 rounded-xl px-4 py-2.5 text-xs border ${
            p.actual.correct ? 'bg-emerald-950/30 border-emerald-800/60 text-emerald-300' : 'bg-rose-950/30 border-rose-800/60 text-rose-300'}`}>
            {p.actual.correct ? <CheckCircle2 className="w-4 h-4" /> : <XCircle className="w-4 h-4" />}
            Реальний результат: <span className="font-mono font-bold">{p.actual.score}</span>
            ({OUTCOME[p.actual.result].short}), модель {p.actual.correct ? 'вгадала' : 'помилилась'}
          </div>
        )}
        {p.split && !['test', 'live'].includes(p.split) && (
          <div className="mb-5 rounded-xl px-4 py-2.5 text-[11px] border border-amber-800/60 bg-amber-950/20 text-amber-300">
            Цей матч входив у навчальну або валідаційну вибірку, тому така перевірка не є об’єктивною.
            Для чесної оцінки обирай матчі з позначкою «тест».
          </div>
        )}
        {p.adjustment && p.base_probs && (
          <div className="mb-5 rounded-xl px-4 py-2.5 text-[11px] border border-[#3a1245] bg-[#14001a]/50 text-[#b9a5c3] leading-relaxed">
            Враховано відсутніх гравців: Elo {p.home_team} {sign(p.adjustment.home_elo)}, Elo {p.away_team} {sign(p.adjustment.away_elo)}.
            Без цього було б: П1 {pct(p.base_probs[0], 0)} · X {pct(p.base_probs[1], 0)} · П2 {pct(p.base_probs[2], 0)}.
            <span className="text-[#9a82a5]"> Поправка евристична, на історичних даних не перевірялась.</span>
          </div>
        )}
        {p.data_until && (
          <div className="mb-4 text-[10px] text-[#9a82a5] font-mono">Статистика форми й Elo актуальна на {fmtDate(p.data_until)}</div>
        )}
        <ProbabilityPanel p={p} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <TeamCompare p={p} />
        <ScoreList p={p} />
      </div>
      <ScoreHeatmap p={p} />
      {live && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <SquadPanel team={p.home_team} accent="emerald" />
          <SquadPanel team={p.away_team} accent="cyan" />
        </div>
      )}
      <ShapChart p={p} />
    </div>
  );
}

function MetricsTab({ data }: { data: MetricsResponse | null }) {
  if (!data) return <Spinner text="Завантаження метрик..." />;
  const ours = data.models.filter((m) => m.kind === 'ours');
  const best = [...ours].sort((a, b) => a.log_loss - b.log_loss)[0];
  const bench = data.models.find((m) => m.kind === 'benchmark');
  const chart = [...data.models].sort((a, b) => a.log_loss - b.log_loss);
  const barColor = (k: string) => (k === 'ours' ? '#e90052' : k === 'benchmark' ? '#f59e0b' : '#7a5a86');

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      <div className="text-xs text-[#b9a5c3]">
        Тестовий період: <span className="font-mono text-[#e8dff0]">{fmtDate(data.test_from)} – {fmtDate(data.test_to)}</span>,{' '}
        {data.test_samples} матчів (модель їх не бачила під час навчання). Train: {data.train_samples}, Val: {data.val_samples}.
      </div>

      {best && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className={`${card} p-6`}>
            <div className="text-xs text-[#b9a5c3] mb-1">Accuracy 1X2</div>
            <div className="text-4xl font-black text-emerald-400 font-mono">{best.accuracy}%</div>
            <div className="text-[11px] text-[#9a82a5] mt-2">{best.name}</div>
          </div>
          <div className={`${card} p-6`}>
            <div className="text-xs text-[#b9a5c3] mb-1">Log Loss (менше = краще)</div>
            <div className="text-4xl font-black text-cyan-400 font-mono">{best.log_loss}</div>
            {bench && <div className="text-[11px] text-[#9a82a5] mt-2">Букмекери: {bench.log_loss}</div>}
          </div>
          <div className={`${card} p-6`}>
            <div className="text-xs text-[#b9a5c3] mb-1">RPS (менше = краще)</div>
            <div className="text-4xl font-black text-[#ff6ba3] font-mono">{best.rps}</div>
            {bench && <div className="text-[11px] text-[#9a82a5] mt-2">Букмекери: {bench.rps}</div>}
          </div>
        </div>
      )}

      <div className={`${card} p-6`}>
        <h3 className="text-sm font-bold text-[#f5eff9] mb-4 flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-[#ff6ba3]" /> Log Loss моделей (шкала не з нуля)
        </h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chart} layout="vertical" margin={{ left: 0, right: 16 }}>
              <XAxis type="number" domain={['dataMin - 0.02', 'dataMax + 0.01']} tick={{ fill: '#9a82a5', fontSize: 10 }} />
              <YAxis dataKey="name" type="category" width={220} tick={{ fill: '#b9a5c3', fontSize: 10 }} />
              <Tooltip contentStyle={{ backgroundColor: '#230029', borderColor: '#3a1245', borderRadius: 8, fontSize: 11 }} />
              <Bar dataKey="log_loss" radius={[0, 4, 4, 0]}>
                {chart.map((m) => <Cell key={m.key} fill={barColor(m.kind)} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className={`${card} p-6 overflow-x-auto`}>
        <table className="w-full text-left text-xs text-[#d5c8dc]">
          <thead className="text-[#b9a5c3] uppercase font-mono">
            <tr>
              <th className="p-3">Модель</th><th className="p-3">Accuracy</th>
              <th className="p-3">Log Loss</th><th className="p-3">RPS</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#3a1245]/60">
            {data.models.map((m) => (
              <tr key={m.key} className={m.kind === 'ours' ? 'bg-[#3a0820]/40 text-white font-semibold' : m.kind === 'benchmark' ? 'text-amber-300' : ''}>
                <td className="p-3">{m.name}</td>
                <td className="p-3 font-mono">{m.accuracy}%</td>
                <td className="p-3 font-mono">{m.log_loss}</td>
                <td className="p-3 font-mono">{m.rps}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="text-[11px] text-[#9a82a5] mt-4 leading-relaxed">
          На вибірці такого розміру похибка accuracy становить близько ±2.5 п.в., тому малі різниці між моделями
          статистично незначущі. Основними метриками є Log Loss та RPS, бо вони оцінюють якість самих ймовірностей.
        </p>
      </div>
    </div>
  );
}

/* ---------- Головний компонент ---------- */
const fmtKickoff = (iso: string | null) =>
  iso ? new Date(iso).toLocaleString('uk-UA', { weekday: 'short', day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' }) : 'час уточнюється';

function MiniBar({ f }: { f: Fixture }) {
  const v = [f.prob_home_win ?? 0, f.prob_draw ?? 0, f.prob_away_win ?? 0];
  return (
    <div className="mt-2">
      <div className="h-1.5 flex gap-0.5 rounded-full overflow-hidden bg-[#14001a]">
        <div className="bg-emerald-500" style={{ width: `${v[0] * 100}%` }} />
        <div className="bg-amber-500" style={{ width: `${v[1] * 100}%` }} />
        <div className="bg-cyan-500" style={{ width: `${v[2] * 100}%` }} />
      </div>
      <div className="flex justify-between mt-1 text-[10px] font-mono text-[#b9a5c3]">
        <span>П1 {pct(v[0], 0)}</span><span>X {pct(v[1], 0)}</span><span>П2 {pct(v[2], 0)}</span>
      </div>
    </div>
  );
}

const TABS = [
  { id: 'predictor', label: 'Прогноз', Icon: Activity },
  { id: 'table', label: 'Таблиця', Icon: Trophy },
  { id: 'injuries', label: 'Травми', Icon: HeartPulse },
  { id: 'simulation', label: 'Симуляція', Icon: Shuffle },
  { id: 'analytics', label: 'Якість моделі', Icon: TrendingUp },
] as const;
type TabId = (typeof TABS)[number]['id'];

export default function App() {
  const [tab, setTab] = useState<TabId>('predictor');
  const [mode, setMode] = useState<'custom' | 'round' | 'history'>('custom');
  const [teams, setTeams] = useState<TeamOption[]>([]);
  const [matches, setMatches] = useState<Match[]>([]);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [round, setRound] = useState<NextRound | null>(null);
  const [event, setEvent] = useState<number | null>(null);
  const [roundError, setRoundError] = useState<string | null>(null);
  const [missing, setMissing] = useState<string[]>([]);
  const [home, setHome] = useState('');
  const [away, setAway] = useState('');
  const [query, setQuery] = useState('');
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const reqId = useRef(0);

  const run = useCallback(async (fn: () => Promise<Prediction>) => {
    const id = ++reqId.current;
    setLoading(true);
    setError(null);
    try {
      const res = await fn();
      if (id === reqId.current) setPrediction(res);
    } catch (e) {
      if (id === reqId.current) setError(errorMessage(e));
    } finally {
      if (id === reqId.current) setLoading(false);
    }
  }, []);

  const predictCustom = useCallback((h: string, a: string) => {
    if (!h || !a) return;
    if (h === a) { setError('Оберіть дві різні команди.'); return; }
    void run(() => api.predictCustom(h, a));
  }, [run]);

  const selectMatch = (id: number) => { setSelectedId(id); void run(() => api.predictMatch(id)); };
  const pickFixture = (f: Fixture) => { setHome(f.home_team); setAway(f.away_team); predictCustom(f.home_team, f.away_team); };

  useEffect(() => {
    (async () => {
      const [t, m, k, h] = await Promise.allSettled([api.teams(), api.recentMatches(), api.metrics(), api.health()]);
      if (m.status === 'fulfilled') setMatches(m.value);
      if (k.status === 'fulfilled') setMetrics(k.value);
      if (h.status === 'fulfilled') setMissing(h.value.missing_teams);
      if (t.status === 'fulfilled' && t.value.length > 1) {
        const names = t.value.map((x) => x.name);
        const hm = names.includes('Arsenal') ? 'Arsenal' : names[0];
        const aw = names.includes('Chelsea') ? 'Chelsea' : names[1];
        setTeams(t.value); setHome(hm); setAway(aw);
        predictCustom(hm, aw);
      } else if (t.status === 'rejected') {
        setError(errorMessage(t.reason));
      }
    })();
  }, [predictCustom]);

  useEffect(() => {
    let alive = true;
    setRoundError(null);
    api.round(event).then((r) => alive && setRound(r)).catch((e) => alive && setRoundError(errorMessage(e)));
    return () => { alive = false; };
  }, [event]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return matches.filter((m) => m.home_team.toLowerCase().includes(q) || m.away_team.toLowerCase().includes(q));
  }, [matches, query]);

  const tabBtn = (active: boolean) =>
    `px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-colors ${
      active ? 'bg-[#e90052] text-white' : 'text-[#b9a5c3] hover:text-[#e8dff0]'}`;
  const modeBtn = (active: boolean) =>
    `py-1.5 px-2 rounded-lg text-[11px] font-semibold flex items-center justify-center gap-1.5 transition-colors ${
      active ? 'bg-[#e90052] text-white' : 'text-[#b9a5c3] hover:text-[#e8dff0]'}`;
  const goRound = (d: number) => setEvent(Math.min(38, Math.max(1, (round?.event ?? 1) + d)));

  return (
    <div className="min-h-screen text-[#f5eff9] font-sans flex flex-col">
      <header className="sticky top-0 z-50 backdrop-blur-md px-6 py-4">
        <div className="max-w-7xl mx-auto flex flex-col lg:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-[#00ff85] rounded-xl shadow-lg shadow-[#00ff85]/20"><Trophy className="w-6 h-6 text-[#38003c]" /></div>
            <div>
              <h1 className="text-2xl text-white leading-none">EPL Match Predictor</h1>
              <p className="text-xs text-[#d5c8dc] mt-1">Ймовірнісні прогнози англійської Прем’єр-ліги з поясненнями</p>
            </div>
          </div>
          <nav className="flex flex-wrap gap-1.5 bg-[#14001a]/60 p-1.5 rounded-xl border border-[#3a1245]">
            {TABS.map(({ id, label, Icon }) => (
              <button key={id} onClick={() => setTab(id)} className={tabBtn(tab === id)}><Icon className="w-4 h-4" /> {label}</button>
            ))}
          </nav>
        </div>
      </header>

      <main className="max-w-7xl w-full mx-auto p-6 flex-1">
        {error && <div className="mb-6 bg-rose-950/50 border border-rose-800 text-rose-200 px-4 py-3 rounded-xl text-xs">{error}</div>}
        {missing.length > 0 && (
          <div className="mb-6 bg-amber-950/30 border border-amber-800/60 text-amber-200 px-4 py-3 rounded-xl text-xs">
            У БД немає даних для: {missing.join(', ')}. Запустіть <span className="font-mono">python -m src.data.sync_football_data</span>,
            потім build_features і train.
          </div>
        )}

        {tab === 'table' && <Standings />}
        {tab === 'injuries' && <Injuries />}
        {tab === 'simulation' && <Simulation />}
        {tab === 'analytics' && (<><MetricsTab data={metrics} /><CalibrationChart /><LiveBacktest /></>)}

        {tab === 'predictor' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            <aside className={`lg:col-span-4 ${card} p-5 flex flex-col lg:h-[820px]`}>
              <div className="grid grid-cols-3 gap-1.5 mb-4 bg-[#14001a] p-1 rounded-xl border border-[#3a1245]/80">
                <button onClick={() => setMode('custom')} className={modeBtn(mode === 'custom')}><Sliders className="w-3.5 h-3.5" /> Свій матч</button>
                <button onClick={() => setMode('round')} className={modeBtn(mode === 'round')}><CalendarDays className="w-3.5 h-3.5" /> Тур</button>
                <button onClick={() => setMode('history')} className={modeBtn(mode === 'history')}><History className="w-3.5 h-3.5" /> Архів</button>
              </div>

              {mode === 'custom' && (
                <div className="space-y-4">
                  <div className="bg-[#14001a]/60 p-4 rounded-xl border border-[#3a1245] space-y-3">
                    <label className="block text-[11px] text-[#b9a5c3] font-medium">Господарі</label>
                    <CustomSelect value={home} onChange={setHome} options={teams} disabledOption={away} />
                    <div className="flex justify-center py-0.5">
                      <button onClick={() => { setHome(away); setAway(home); }} title="Поміняти місцями"
                        className="p-1.5 rounded-lg text-[#9a82a5] hover:text-[#ff6ba3] hover:bg-[#3a1245] transition-colors">
                        <ArrowLeftRight className="w-4 h-4 rotate-90" />
                      </button>
                    </div>
                    <label className="block text-[11px] text-[#b9a5c3] font-medium">Гості</label>
                    <CustomSelect value={away} onChange={setAway} options={teams} disabledOption={home} />
                    <button onClick={() => predictCustom(home, away)} disabled={loading || !home}
                      className="w-full mt-2 py-2.5 bg-gradient-to-r from-[#e90052] to-cyan-600 hover:from-[#ff2e7e] hover:to-cyan-500 disabled:opacity-50 text-white rounded-xl text-xs font-bold shadow-lg shadow-[#e90052]/20 transition-all flex items-center justify-center gap-2">
                      <Zap className="w-4 h-4" /> Спрогнозувати
                    </button>
                  </div>
                  <div className="p-4 bg-[#3a0820]/20 border border-[#5c0a30]/40 rounded-xl text-[11px] text-[#b9a5c3] leading-relaxed">
                    Прогноз завжди враховує травми та дискваліфікації (дані FPL). Гравці з шансом зіграти ≥75% вважаються доступними.
                    Модель: Elo, ковзна xG-форма, ансамбль CatBoost + LightGBM + логістична регресія.
                  </div>
                </div>
              )}

              {mode === 'round' && (
                <div className="flex flex-col flex-1 min-h-0">
                  <div className="flex items-center justify-between mb-3">
                    <button onClick={() => goRound(-1)} className="p-1.5 rounded-lg text-[#b9a5c3] hover:bg-[#3a1245]"><ChevronLeft className="w-4 h-4" /></button>
                    <span className="text-xs font-bold uppercase tracking-wider text-[#e8dff0]">Тур {round?.event ?? '…'}</span>
                    <button onClick={() => goRound(1)} className="p-1.5 rounded-lg text-[#b9a5c3] hover:bg-[#3a1245]"><ChevronRight className="w-4 h-4" /></button>
                  </div>
                  <div className="space-y-2 overflow-y-auto pr-1 flex-1">
                    {roundError && <div className="text-xs text-amber-300">{roundError}</div>}
                    {!round && !roundError && <div className="text-xs text-[#9a82a5] animate-pulse">Завантаження календаря...</div>}
                    {round?.fixtures.map((f) => {
                      const ok = f.prob_home_win !== null;
                      const active = prediction?.home_team === f.home_team && prediction?.away_team === f.away_team;
                      return (
                        <button key={f.home_team + f.away_team} disabled={!ok} onClick={() => pickFixture(f)}
                          className={`w-full text-left p-3 rounded-xl border transition-colors ${
                            active ? 'bg-[#3a0820]/70 border-[#ff2e7e]' : ok ? 'bg-[#14001a]/50 border-[#3a1245]/70 hover:border-[#52275f]' : 'bg-[#14001a]/30 border-[#3a1245]/50 opacity-70'}`}>
                          <div className="text-[10px] text-[#9a82a5] font-mono mb-2">{fmtKickoff(f.kickoff)}</div>
                          <div className="flex items-center justify-between gap-2 text-xs font-medium">
                            <span className="flex items-center gap-2 min-w-0"><TeamBadge src={f.home_crest} name={f.home_team} size={22} /><span className="truncate">{f.home_team}</span></span>
                            <span className="text-[#7a5a86] text-[10px]">vs</span>
                            <span className="flex items-center gap-2 min-w-0 justify-end"><span className="truncate">{f.away_team}</span><TeamBadge src={f.away_crest} name={f.away_team} size={22} /></span>
                          </div>
                          {ok && <MiniBar f={f} />}
                          {f.note && <div className="text-[10px] text-amber-400 mt-1.5">{f.note}</div>}
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}

              {mode === 'history' && (
                <>
                  <div className="relative mb-4">
                    <Search className="w-4 h-4 absolute left-3 top-2.5 text-[#9a82a5]" />
                    <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Пошук команди..."
                      className="w-full bg-[#14001a] border border-[#3a1245] rounded-xl pl-9 pr-4 py-2 text-xs text-[#e8dff0] placeholder-[#9a82a5] focus:outline-none focus:border-[#ff2e7e]" />
                  </div>
                  <div className="space-y-2 overflow-y-auto pr-1 flex-1 max-h-[560px] lg:max-h-none">
                    {filtered.map((m) => (
                      <button key={m.id} onClick={() => selectMatch(m.id)}
                        className={`w-full text-left p-3 rounded-xl border transition-colors ${
                          selectedId === m.id ? 'bg-[#3a0820]/70 border-[#ff2e7e] text-white' : 'bg-[#14001a]/50 border-[#3a1245]/70 hover:border-[#52275f] text-[#d5c8dc]'}`}>
                        <div className="flex justify-between items-center text-[10px] text-[#9a82a5] mb-1.5 font-mono">
                          <span>{fmtDate(m.match_date)}</span>
                          <span className={`px-1.5 py-0.5 rounded ${['test', 'live'].includes(m.split) ? 'bg-[#5c0a30]/60 text-[#ff9ec3]' : 'bg-[#3a1245] text-[#b9a5c3]'}`}>
                            {['test', 'live'].includes(m.split) ? 'тест' : 'навч.'}
                          </span>
                        </div>
                        <div className="flex justify-between items-center text-xs font-medium gap-2">
                          <span className="truncate">{m.home_team} <span className="text-[#9a82a5]">vs</span> {m.away_team}</span>
                          <span className={`font-mono px-2 py-0.5 rounded font-bold shrink-0 ${OUTCOME[m.result_1x2].badge}`}>{m.home_score}:{m.away_score}</span>
                        </div>
                      </button>
                    ))}
                    {filtered.length === 0 && <div className="text-xs text-[#9a82a5] text-center py-8">Нічого не знайдено</div>}
                  </div>
                </>
              )}
            </aside>

            <section className="lg:col-span-8">
              {loading ? <Spinner text="Обчислення ймовірностей..." />
                : prediction ? <PredictionView p={prediction} />
                : <div className={`${card} p-16 text-center text-xs text-[#9a82a5]`}>Оберіть матч, щоб побачити прогноз</div>}
            </section>
          </div>
        )}
      </main>

      <footer className="border-t border-[#3a1245] px-6 py-4 text-center text-[11px] text-[#9a82a5]">
        Навчальний проєкт. Прогнози мають імовірнісний характер і не є порадою для ставок. Емблеми й фото належать клубам та АПЛ.
      </footer>
    </div>
  );
}