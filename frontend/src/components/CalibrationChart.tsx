import { useEffect, useState } from 'react';
import { CartesianGrid, Legend, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../api';
import type { Calibration } from '../types';

const COL = { H: '#10b981', D: '#f59e0b', A: '#22d3ee' } as const;
const NAME = { H: 'П1', D: 'Нічия', A: 'П2' } as const;

export default function CalibrationChart() {
  const [cal, setCal] = useState<Calibration | null>(null);
  useEffect(() => { api.calibration('market').then(setCal).catch(() => setCal(null)); }, []);
  if (!cal) return null;
  const tick = { fill: '#b9a5c3', fontSize: 10 };
  return (
    <div className="max-w-5xl mx-auto mt-8 bg-slate-900/70 border border-slate-800 rounded-2xl p-6">
      <h3 className="text-sm font-bold text-slate-100">Калібрування ймовірностей (тест, {cal.n} матчів)</h3>
      <p className="text-[11px] text-slate-400 mt-1 mb-3">
        Чим ближче точки до діагоналі, тим чесніші ймовірності. Середня похибка калібрування (ECE): {cal.ece.toFixed(3)}.
      </p>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ left: 0, right: 16, top: 8, bottom: 8 }}>
            <CartesianGrid stroke="#3a1245" strokeDasharray="3 3" />
            <XAxis type="number" dataKey="pred" domain={[0, 1]} tick={tick} name="Прогноз" />
            <YAxis type="number" dataKey="obs" domain={[0, 1]} tick={tick} name="Фактично" />
            <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#9a82a5" strokeDasharray="5 5" />
            <Tooltip contentStyle={{ backgroundColor: '#230029', borderColor: '#3a1245', borderRadius: 8, fontSize: 11 }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {(['H', 'D', 'A'] as const).map((k) => (
              <Scatter key={k} name={NAME[k]} data={cal.classes[k]} fill={COL[k]} line />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}