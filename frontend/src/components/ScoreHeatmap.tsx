import type { Prediction } from '../types';

const pct = (x: number, d = 1) => `${(x * 100).toFixed(d)}%`;

export default function ScoreHeatmap({ p }: { p: Prediction }) {
  const m = p.score_matrix;
  const max = Math.max(...m.flat());
  const idx = [0, 1, 2, 3, 4, 5];
  return (
    <div className="bg-[#230029]/70 border border-[#3a1245] rounded-2xl p-5">
      <h3 className="text-sm font-bold text-[#f5eff9] mb-1">Матриця рахунків</h3>
      <p className="text-[11px] text-[#b9a5c3] mb-4">
        Рядки: голи {p.home_team}, стовпці: голи {p.away_team}. Форму розподілу дає модель Діксона-Коулза,
        загальні П1/Х/П2 вирівняні під ML-прогноз.
      </p>
      <div className="grid gap-1" style={{ gridTemplateColumns: 'auto repeat(6, 1fr)' }}>
        <div />
        {idx.map((a) => <div key={a} className="text-center text-[10px] font-mono text-cyan-400">{a}</div>)}
        {idx.map((h) => (
          <div key={h} className="contents">
            <div className="flex items-center justify-center pr-1.5 text-[10px] font-mono text-emerald-400">{h}</div>
            {idx.map((a) => {
              const v = m[h][a];
              return (
                <div key={a} title={`${h}:${a} — ${pct(v)}`}
                  style={{ backgroundColor: `rgba(99,102,241,${0.06 + 0.94 * (v / max)})` }}
                  className="aspect-square rounded flex items-center justify-center text-[9px] font-mono text-white/90">
                  {v >= 0.012 ? (v * 100).toFixed(1) : ''}
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-3 mt-4 text-center">
        <div className="rounded-xl border border-[#3a1245] bg-[#14001a]/50 p-3">
          <div className="text-[10px] text-[#9a82a5] mb-1">Тотал більше 2.5 / менше 2.5</div>
          <div className="text-sm font-bold font-mono text-[#f5eff9]">{pct(p.over_25, 0)} / {pct(1 - p.over_25, 0)}</div>
        </div>
        <div className="rounded-xl border border-[#3a1245] bg-[#14001a]/50 p-3">
          <div className="text-[10px] text-[#9a82a5] mb-1">Обидві заб’ють: так / ні</div>
          <div className="text-sm font-bold font-mono text-[#f5eff9]">{pct(p.btts, 0)} / {pct(1 - p.btts, 0)}</div>
        </div>
      </div>
    </div>
  );
}