import type { ReactNode } from 'react';
import {
  ChartContainer,
  DataState,
  DataTable,
  DefinitionList,
  EmptyState,
  Panel,
} from '@/components/ui';
import { list, number, object, text, valueAt, type Json } from '@/lib/api';
import { useReport } from '@/lib/queries';
import { fmt, human } from '@/lib/format';
export function ResearchData({
  path,
  children,
}: {
  path: string;
  children: (data: Record<string, Json>) => ReactNode;
}) {
  const query = useReport(path);
  return (
    <DataState query={query}>
      {(report) => {
        const data = object(report).data;
        if (data == null)
          return (
            <EmptyState
              title="Analysis unavailable"
              description={
                typeof object(report).reason === 'string'
                  ? text(object(report).reason)
                  : human(text(object(report).status))
              }
            />
          );
        return children(object(data));
      }}
    </DataState>
  );
}
export function NumericSummary({
  data,
  title = 'Numerical summaries',
}: {
  data: Json | undefined;
  title?: string;
}) {
  return (
    <Panel title={title}>
      <DataTable
        label={title}
        headers={[
          'Feature',
          'Available count',
          'Mean',
          'Median',
          'Variance',
          'Std. deviation',
          'IQR',
        ]}
        rows={Object.entries(object(data)).map(([key, v]) => ({
          id: key,
          cells: [
            human(key),
            fmt(number(valueAt(v, 'count')), 0),
            fmt(number(valueAt(v, 'mean'))),
            fmt(number(valueAt(v, 'median'))),
            fmt(number(valueAt(v, 'variance'))),
            fmt(number(valueAt(v, 'standard_deviation'))),
            fmt(number(valueAt(v, 'iqr'))),
          ],
        }))}
      />
    </Panel>
  );
}
export function GroupDistribution({
  data,
  signal,
  title,
}: {
  data: Json | undefined;
  signal: string;
  title: string;
}) {
  const rows = Object.entries(object(data)).map(([group, v]) => ({
    name: human(group),
    median: number(valueAt(v, signal, 'median')),
    mean: number(valueAt(v, signal, 'mean')),
  }));
  return (
    <ChartContainer
      title={title}
      note="Validation / Development. Score summaries exclude unavailable evidence."
      data={rows}
      x="name"
      series={[
        { key: 'median', label: 'Median' },
        { key: 'mean', label: 'Mean', color: 'var(--slate)' },
      ]}
    />
  );
}
export function CorrelationTable({ data }: { data: Json | undefined }) {
  const pairs = list(valueAt(data, 'pairs'))
    .map(object)
    .sort((a, b) => Math.abs(number(b.spearman) ?? 0) - Math.abs(number(a.spearman) ?? 0))
    .slice(0, 20);
  return (
    <Panel
      title="Correlation evidence"
      note="Training research snapshot. Association does not establish causation."
    >
      <DataTable
        label="Correlations"
        headers={['Feature pair', 'Pearson', 'Spearman', 'Available pairs']}
        rows={pairs.map((p, i) => ({
          id: String(i),
          cells: [
            `${human(text(p.first))} / ${human(text(p.second))}`,
            fmt(number(p.pearson), 4),
            fmt(number(p.spearman), 4),
            fmt(number(p.pair_count), 0),
          ],
        }))}
      />
    </Panel>
  );
}
export function ScalarFacts({ data }: { data: Json | undefined }) {
  return (
    <DefinitionList
      items={Object.entries(object(data))
        .filter(([, v]) => v === null || typeof v !== 'object')
        .map(([key, v]) => [
          human(key),
          v === null
            ? 'Unavailable'
            : typeof v === 'number'
              ? fmt(v, 6)
              : typeof v === 'string'
                ? v
                : JSON.stringify(v),
        ])}
    />
  );
}
