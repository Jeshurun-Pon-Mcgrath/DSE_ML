import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { fmt, pct } from '@/lib/format';
export interface ChartProps {
  data: Record<string, string | number | null>[];
  x: string;
  series: { key: string; label: string; color?: string }[];
  kind?: 'area' | 'bar';
  horizontal?: boolean;
  percent?: boolean;
}
export default function Chart({
  data,
  x,
  series,
  kind = 'bar',
  horizontal = false,
  percent = false,
}: ChartProps) {
  if (!data.length) return <p className="caption">No available observations for this chart.</p>;
  const grid = <CartesianGrid vertical={false} stroke="var(--border)" strokeDasharray="3 4" />;
  const tick = (v: number) =>
    percent
      ? `${Math.round(v * 100)}%`
      : new Intl.NumberFormat('en', { notation: 'compact' }).format(v);
  const axes = horizontal ? (
    <>
      <XAxis type="number" tickFormatter={tick} />
      <YAxis type="category" dataKey={x} width={160} tick={{ fontSize: 11 }} />
    </>
  ) : (
    <>
      <XAxis dataKey={x} tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
      <YAxis tickFormatter={tick} tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
    </>
  );
  return (
    <div
      className="chart"
      role="img"
      aria-label={`Chart of ${series.map((s) => s.label).join(', ')}`}
    >
      <ResponsiveContainer width="100%" height="100%">
        {kind === 'area' ? (
          <AreaChart
            accessibilityLayer
            data={data}
            margin={{ top: 10, right: 10, left: 0, bottom: 5 }}
          >
            {grid}
            {axes}
            <Tooltip
              formatter={(v) =>
                typeof v === 'number' ? (percent ? pct(v) : fmt(v, 4)) : String(v ?? 'Unavailable')
              }
            />
            <Legend />
            {series.map((s, i) => (
              <Area
                key={s.key}
                type="monotone"
                dataKey={s.key}
                name={s.label}
                stroke={s.color ?? (i ? 'var(--amber)' : 'var(--teal)')}
                fill={s.color ?? 'var(--teal)'}
                fillOpacity={0.08}
                strokeWidth={2}
                isAnimationActive={false}
              />
            ))}
          </AreaChart>
        ) : (
          <BarChart
            accessibilityLayer
            data={data}
            layout={horizontal ? 'vertical' : 'horizontal'}
            margin={{ top: 10, right: 10, left: 0, bottom: 5 }}
          >
            {grid}
            {axes}
            <Tooltip
              formatter={(v) =>
                typeof v === 'number' ? (percent ? pct(v) : fmt(v, 4)) : String(v ?? 'Unavailable')
              }
            />
            <Legend />
            {series.map((s, i) => (
              <Bar
                key={s.key}
                dataKey={s.key}
                name={s.label}
                fill={s.color ?? (i ? 'var(--amber)' : 'var(--teal)')}
                maxBarSize={32}
                radius={[2, 2, 0, 0]}
                isAnimationActive={false}
              />
            ))}
          </BarChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}
